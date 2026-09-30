import math
import torch

from torch import Tensor, nn
from torch_geometric.utils import degree
from typing import Optional, Literal, Union, Tuple
from torch._higher_order_ops import while_loop

from .attention import Denoiser, SamplingState


class Scheduler(object):
    def __init__(self, SR_max: float=1., SR_min: float=1e-2, scheme: str='cosine'):
        self.SR_max = SR_max
        self.SR_min = SR_min
        try:
            self.scheme = getattr(self, scheme)
        except AttributeError:
            raise NotImplementedError(f"Diffusion scheme {scheme} is not implemented.")

    def __call__(self, T: Tensor):
        return self.scheme(T)

    def cosine(self, T: Tensor):
        alpha = math.acos(self.SR_max) + (
            T * (math.acos(self.SR_min) - math.acos(self.SR_max)) )

        beta  = 2 * (math.acos(self.SR_min)
            - math.acos(self.SR_max)) * torch.tan(alpha)

        return torch.cos(alpha).view(-1,1), torch.sin(alpha).view(-1,1), beta


class TimeGrid(nn.Module):
    r"""Discretisation of the reverse diffusion time used by :meth:`NeutrinoDiffusion.solver`.

    With :math:`N` sampling steps, the solver visits :math:`1 = T_0 > T_1 > \dots > T_N = 0`
    with :math:`T_i = w(1 - i/N)`, where :math:`w` is a monotonic map of :math:`[0, 1]` onto
    itself. The density of steps at :math:`T` is :math:`\rho(T) = \mathrm{d}w^{-1}/\mathrm{d}T`,
    so a scheme spends more of a fixed step budget where :math:`\rho` is large.

    Schemes:

    - ``uniform``: :math:`T = u`, i.e. :math:`\Delta T = 1/N`. The solver then decrements
      :math:`T` by :math:`\Delta T` at every step, exactly as before this option existed.
    - ``power``: :math:`T = u^{\rho}`. :obj:`rho` > 1 refines the steps near :math:`T = 0`
      (the data end), :obj:`rho` < 1 near :math:`T = 1` (the noise end).
    - ``cosine``: :math:`T = (1 - \cos \pi u) / 2`, refines both ends.
    - ``beta``: :math:`\rho(T) = \mathrm{Beta}(T; a, b)`, the timestep distribution family
      of the adaptive timestep sampler in `arXiv:2411.09998 <https://arxiv.org/abs/2411.09998>`_.
      :obj:`a`, :obj:`b` < 1 refine the ends, > 1 the middle; :obj:`a` = :obj:`b` = 1 is uniform.
    - ``adaptive``: :math:`\rho` is estimated from the trained model itself by
      :meth:`NeutrinoDiffusion.calibrate_time_grid`. A reference solve with
      :obj:`calibration_steps` uniform steps measures the local truncation error density
      :math:`\kappa(T)` of the DDIM step, and :math:`\rho \propto \kappa^{\gamma}` is mixed with a
      uniform density (weight :obj:`uniform_mix`). :math:`\gamma = 1/2` minimises the
      leading-order global error for a fixed number of steps. Calibration runs on the first
      batch passed to the solver, unless the grid was calibrated before.

    ``beta`` and ``adaptive`` store :math:`w` as a lookup table, a non-persistent buffer
    (not saved in checkpoints). All schemes support a tensor number of steps (ONNX export).

    Args:
        scheme (str): one of ``uniform``, ``power``, ``cosine``, ``beta`` or ``adaptive``.
            (default: :obj:`str`=``uniform``)
        rho (float): exponent of the ``power`` scheme. (default: :obj:`float`=2.)
        a (float): first shape parameter of the ``beta`` scheme. (default: :obj:`float`=1.)
        b (float): second shape parameter of the ``beta`` scheme. (default: :obj:`float`=1.)
        gamma (float): exponent of the ``adaptive`` scheme. (default: :obj:`float`=0.5)
        uniform_mix (float): weight of the uniform density in the ``adaptive`` scheme.
            (default: :obj:`float`=0.25)
        calibration_steps (int): uniform steps of the ``adaptive`` reference solve.
            (default: :obj:`int`=1000)
        calibration_size (int): maximum number of neutrinos used for calibration; whole
            events are kept. (default: :obj:`int`=4096)
    """
    schemes = ('uniform', 'power', 'cosine', 'beta', 'adaptive')
    table_size = 2049

    def __init__(
        self,
        scheme: str = 'uniform',
        rho: float = 2.,
        a: float = 1.,
        b: float = 1.,
        gamma: float = 0.5,
        uniform_mix: float = 0.25,
        calibration_steps: int = 1000,
        calibration_size: int = 4096
    ) -> None:
        super().__init__()
        if scheme not in self.schemes:
            raise NotImplementedError(f"Time grid scheme `{scheme}` is not implemented, choose from {self.schemes}.")
        if rho <= 0 or a <= 0 or b <= 0:
            raise ValueError("`rho`, `a` and `b` must be positive.")
        if not 0. <= uniform_mix <= 1.:
            raise ValueError("`uniform_mix` must be in [0, 1].")
        self.scheme = scheme
        self.rho = float(rho)
        self.a, self.b = float(a), float(b)
        self.gamma = float(gamma)
        self.uniform_mix = float(uniform_mix)
        self.calibration_steps = int(calibration_steps)
        self.calibration_size = int(calibration_size)

        # Lookup table of w(u) at u = k / (table_size - 1), only for `beta` and `adaptive`,
        # so the other schemes add no buffers to the model
        if scheme in ('beta', 'adaptive'):
            self.register_buffer('table', torch.empty(0), persistent=False)
        # Calibrated step density (T, rho(T)) of the `adaptive` scheme, for diagnostics
        self.density = None
        if scheme == 'beta':
            self.table = self._beta_table(self.a, self.b)

    @property
    def needs_calibration(self) -> bool:
        return self.scheme == 'adaptive' and self.table.numel() == 0

    def extra_repr(self) -> str:
        opts = {'power': f", rho={self.rho}", 'beta': f", a={self.a}, b={self.b}",
                'adaptive': f", gamma={self.gamma}, uniform_mix={self.uniform_mix}, "
                            f"calibrated={not self.needs_calibration}"}
        return f"scheme={self.scheme}{opts.get(self.scheme, '')}"

    @classmethod
    def _inverse_cdf(cls, cdf: Tensor, T: Tensor) -> Tensor:
        # Invert an increasing CDF, given at the increasing knots `T`, at `table_size` uniform points
        u = torch.linspace(0., 1., cls.table_size, dtype=torch.float64)
        idx = torch.searchsorted(cdf, u).clamp(1, cdf.numel() - 1)
        x0, x1 = cdf[idx - 1], cdf[idx]
        w = ((u - x0) / (x1 - x0).clamp_min(torch.finfo(torch.float64).tiny)).clamp(0., 1.)
        table = T[idx - 1] + w * (T[idx] - T[idx - 1])
        table[0], table[-1] = 0., 1.
        return table.to(torch.float32)

    @classmethod
    def _beta_table(cls, a: float, b: float, n: int = 20001) -> Tensor:
        # CDF of Beta(a, b), integrated with the midpoint rule after substituting
        # T = (1 - cos(pi s)) / 2, which clusters the nodes at the (possibly singular) ends
        s = torch.linspace(0., 1., n, dtype=torch.float64)
        s_mid = 0.5 * (s[1:] + s[:-1])
        t_mid = 0.5 * (1. - torch.cos(math.pi * s_mid))
        w = t_mid.pow(a - 1) * (1. - t_mid).pow(b - 1) * torch.sin(math.pi * s_mid)
        cdf = torch.cat([w.new_zeros(1), torch.cumsum(w, 0)])
        return cls._inverse_cdf(cdf / cdf[-1], 0.5 * (1. - torch.cos(math.pi * s)))

    def set_density(self, T: Tensor, density: Tensor) -> None:
        r"""Set the step density of the ``adaptive`` scheme.

        Args:
            T (torch.Tensor): increasing knots of shape [M+1] spanning [0, 1].
            density (torch.Tensor): non-negative density in each of the M cells.
        """
        T = T.detach().to('cpu', torch.float64)
        density = density.detach().to('cpu', torch.float64).clamp_min(0.)
        dT = T[1:] - T[:-1]
        norm = (density * dT).sum()
        density = density / norm if norm > 0 else torch.ones_like(density)
        density = (1. - self.uniform_mix) * density + self.uniform_mix
        cdf = torch.cat([density.new_zeros(1), torch.cumsum(density * dT, 0)])
        self.table = self._inverse_cdf(cdf / cdf[-1], T).to(self.table.device)
        self.density = (0.5 * (T[1:] + T[:-1]), density)

    def warp(self, u: Tensor) -> Tensor:
        r"""Map :obj:`u` in [0, 1] to diffusion time :math:`T = w(u)`."""
        if self.scheme == 'uniform':
            return u
        if self.scheme == 'power':
            return u.pow(self.rho)
        if self.scheme == 'cosine':
            return 0.5 * (1. - torch.cos(math.pi * u))
        if self.needs_calibration:
            raise RuntimeError("The `adaptive` time grid is not calibrated, see `NeutrinoDiffusion.calibrate_time_grid`.")
        # Linear interpolation of the lookup table
        x = u * (self.table_size - 1)
        i0 = torch.floor(x).clamp(0, self.table_size - 2)
        w = x - i0
        i0 = i0.to(torch.int64).reshape(-1)
        T0 = self.table.gather(0, i0).reshape(u.shape)
        T1 = self.table.gather(0, i0 + 1).reshape(u.shape)
        return T0 * (1. - w) + T1 * w

    def forward(self, i: Tensor, num_steps: Tensor) -> Tensor:
        r"""The timestep :math:`T_i` of a solve with :obj:`num_steps` steps, in float32.

        Args:
            i (torch.Tensor): step index (or indices), from 0 to :obj:`num_steps`.
            num_steps (torch.Tensor): number of steps, a scalar tensor.

        :rtype: :class:`torch.Tensor`
        """
        return self.warp(1. - i.to(torch.float32) / num_steps.to(torch.float32))

    def grid(self, num_steps: int) -> Tensor:
        r"""All :obj:`num_steps` + 1 timesteps, decreasing from 1 to 0."""
        device = self.table.device if hasattr(self, 'table') else None
        return self(torch.arange(num_steps + 1, device=device), torch.tensor(num_steps, device=device))


class NeutrinoDiffusion(nn.Module):
    def __init__(
        self,
        d_ctx: int,
        d_embed: int,
        d_target: int,
        num_heads: int,
        num_message_steps: int,
        num_dit_blocks: int = 4,
        noise_distribution: Literal["gaussian", "uniform"] = "uniform",
        num_sampling_steps: int = 1000,
        sampling_schedule: Optional[dict] = None
    ) -> None:
        super().__init__()
        self.d_ctx = d_ctx
        self.d_embed = d_embed
        self.d_target = d_target
        self.num_message_steps  = num_message_steps
        self.num_sampling_steps = num_sampling_steps
        self._use_gaussian_noise = (noise_distribution == "gaussian")
        self.scheduler = Scheduler()
        self.time_grid = TimeGrid(**dict(sampling_schedule or {}))

        self.Denoiser = Denoiser(
            d_embed=d_embed,
            depth=num_dit_blocks,
            num_heads=num_heads,
            d_x=d_target,
            d_ctx=(d_ctx*num_message_steps),
            mlp_ratio=4.,
            dropout=0.
        )

    def sample_noise(self, n: int, device=None) -> Tensor:
        r"""Sample the initial noise of :meth:`solver`, ~N(0,1) or ~U(0,1).

        Args:
            n (int): number of neutrinos.
            device (torch.device, optional): device of the output. (default :obj:`None`)

        :rtype: :class:`torch.Tensor`
        """
        if self._use_gaussian_noise:
            return torch.randn((n, self.d_target), device=device)
        return torch.rand((n, self.d_target), device=device)

    @torch.no_grad()
    def solver(
            self,
            ctx: Tensor,
            nu_batch: Tensor,
            noise: Optional[Tensor] = None,
            num_steps: Optional[Union[int, Tensor]] = None
        ) -> Tensor:
        r"""Sample neutrinos from noise with deterministic (DDIM) sampling, stepping
        through the timesteps of :attr:`time_grid`.

        Args:
            ctx (torch.Tensor): context vectors of shape [N, `d_ctx`].
            nu_batch (torch.Tensor): sorted event index of each neutrino, of shape [N].
            noise (torch.Tensor, optional): initial noise of shape [N, `d_target`],
                sampled with :meth:`sample_noise` if not given. (default :obj:`None`)
            num_steps (int or torch.Tensor, optional): number of sampling steps. A tensor
                is exported to ONNX as a single `Loop` with a runtime number of steps.
                (default :obj:`None`=`num_sampling_steps`)

        :rtype: :class:`torch.Tensor`
        """
        if noise is None:
            noise = self.sample_noise(ctx.size(0), device=ctx.device)
        torch._check(noise.size(0) == ctx.size(0))
        if num_steps is None:
            num_steps = self.num_sampling_steps

        if self.time_grid.needs_calibration:
            if torch.compiler.is_exporting():
                raise RuntimeError("Calibrate the `adaptive` time grid before export, "
                                   "see `NeutrinoDiffusion.calibrate_time_grid`.")
            # No neutrinos to calibrate on, nor to sample
            if ctx.size(0) == 0:
                return noise
            self.calibrate_time_grid(ctx, nu_batch)
            print(f"Calibrated time grid ({self.time_grid.extra_repr()}) "
                  f"with {self.time_grid.calibration_steps} uniform steps.")

        # Tensors shared by all steps
        state = self.Denoiser.prepare_sampling(ctx, nu_batch)

        # Starting from pure noise and removing noise iteratively:
        D = self.Denoiser.to_dense(noise, state)

        if self.time_grid.scheme == 'uniform':
            # Constant steps, the same as before `sampling_schedule` was introduced
            # Timesteps
            # Kept in float32: `T` is decremented `num_steps` times and
            # would drift under reduced-precision dtypes
            T = torch.ones((), dtype=torch.float32, device=ctx.device)

            if isinstance(num_steps, Tensor):
                num_steps = num_steps.reshape(()).to(torch.int64)
                dT = 1. / num_steps.to(torch.float32)

                def cond_fn(i, T, D):
                    return i < num_steps

                def body_fn(i, T, D):
                    D = self._step(T, T - dT, D, state)
                    return i + 1, T - dT, D

                i = torch.zeros((), dtype=torch.int64, device=ctx.device)
                _, _, D = while_loop(cond_fn, body_fn, (i, T, D))
            else:
                dT = 1 / num_steps
                for _ in range(num_steps):
                    D = self._step(T, T - dT, D, state)
                    T = T - dT
            return self.Denoiser.from_dense(D, state)

        # Timesteps 1 = T_0 > T_1 > ... > T_N = 0 of `time_grid`, each computed
        # from its index (in float32, rather than accumulated) so they do not drift
        if isinstance(num_steps, Tensor):
            num_steps = num_steps.reshape(()).to(torch.int64)

            def cond_fn(i, D):
                return i < num_steps

            def body_fn(i, D):
                D = self._step(self.time_grid(i, num_steps), self.time_grid(i + 1, num_steps), D, state)
                return i + 1, D

            i = torch.zeros((), dtype=torch.int64, device=ctx.device)
            _, D = while_loop(cond_fn, body_fn, (i, D))
        else:
            grid = self.time_grid(torch.arange(num_steps + 1, device=ctx.device),
                                  torch.tensor(num_steps, device=ctx.device))
            for i in range(num_steps):
                D = self._step(grid[i], grid[i + 1], D, state)

        return self.Denoiser.from_dense(D, state)

    def _step(self, T_i: Tensor, T_j: Tensor, D: Tensor, state: SamplingState,
              return_preds: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        r"""One sampling step from :obj:`T_i` to :obj:`T_j`.

        Args:
            T_i (torch.Tensor): current timestep, a scalar tensor.
            T_j (torch.Tensor): next timestep, a scalar tensor.
            D (torch.Tensor): noised data in the dense layout, of shape [B, L, `d_target`].
            state (SamplingState): from :meth:`Denoiser.prepare_sampling`.
            return_preds (bool): also return the data and noise predicted at :obj:`T_i`.
                (default :obj:`False`)

        :rtype: :class:`torch.Tensor`
        """
        # Diffusion Scheduler
        signal_rate_i, noise_rate_i, _ = self.scheduler(T_i)
        signal_rate_j, noise_rate_j, _ = self.scheduler(T_j)
        # Denoising model
        noise_pred = self.Denoiser.sample_step(D, T_i, state)
        data_pred  = (D - noise_rate_i * noise_pred) / signal_rate_i

        D = signal_rate_j * data_pred + noise_rate_j * noise_pred
        # Reset padded positions
        D = D * state.keep
        return (D, data_pred, noise_pred) if return_preds else D

    @torch.no_grad()
    def calibrate_time_grid(self, ctx: Tensor, nu_batch: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        r"""Calibrate the ``adaptive`` time grid on a batch.

        Solves the reverse diffusion with :obj:`time_grid.calibration_steps` uniform steps and
        estimates the local truncation error density of the DDIM step along the trajectory,

        .. math::
            \kappa(T) = \big\langle \| -\sin\phi\, \partial_T \hat{x}_0 + \cos\phi\, \partial_T \hat{\epsilon} \| \big\rangle,

        where :math:`\cos\phi` and :math:`\sin\phi` are the signal and noise rates. The DDIM
        step is exact when :math:`\hat{x}_0` and :math:`\hat{\epsilon}` do not change, and its
        error over a step :math:`h` is :math:`\approx h^2 \kappa / 2`. The step density is set to
        :math:`\rho \propto \kappa^{\gamma}`.

        Args:
            ctx (torch.Tensor): context vectors of shape [N, `d_ctx`].
            nu_batch (torch.Tensor): sorted event index of each neutrino, of shape [N].
            noise (torch.Tensor, optional): initial noise. (default :obj:`None`)

        :rtype: :class:`torch.Tensor` the estimated :math:`\kappa` in each of the uniform cells,
            ordered in increasing :math:`T`.
        """
        grid_cfg = self.time_grid
        if ctx.size(0) == 0:
            raise ValueError("No neutrinos to calibrate the time grid on.")
        # Keep whole events, up to `calibration_size` neutrinos
        n = min(ctx.size(0), grid_cfg.calibration_size)
        if n < ctx.size(0):
            keep = nu_batch < nu_batch[n]
            if not keep.any():
                keep = nu_batch == nu_batch[0]
            ctx, nu_batch = ctx[keep], nu_batch[keep]
            noise = None if noise is None else noise[keep]
        if noise is None:
            noise = self.sample_noise(ctx.size(0), device=ctx.device)

        state = self.Denoiser.prepare_sampling(ctx, nu_batch)
        D = self.Denoiser.to_dense(noise, state)
        real = state.keep[..., 0]

        M = grid_cfg.calibration_steps
        grid = 1. - torch.arange(M + 1, dtype=torch.float32, device=ctx.device) / M
        kappa, prev = [], None
        for k in range(M):
            D, data_pred, noise_pred = self._step(grid[k], grid[k+1], D, state, return_preds=True)
            if prev is not None:
                # Finite differences between consecutive predictions, at the cell midpoint
                signal_rate, noise_rate, _ = self.scheduler(0.5 * (grid[k-1] + grid[k]))
                diff = -noise_rate * (data_pred - prev[0]) + signal_rate * (noise_pred - prev[1])
                kappa.append((diff.float().norm(dim=-1) * real).sum() / real.sum() * M)
            prev = (data_pred, noise_pred)
        # M - 1 estimates, at T in (1/M, 1); extend to the M cells of [0, 1], in increasing T
        kappa = torch.stack(kappa)
        kappa = torch.cat([kappa, kappa[-1:]]).flip(0)

        # Light smoothing
        width = max(1, M // 100) | 1
        kernel = torch.ones(1, 1, width, device=kappa.device) / width
        kappa = torch.nn.functional.conv1d(
            torch.nn.functional.pad(kappa.view(1, 1, -1), (width // 2, width // 2), mode='replicate'),
            kernel).view(-1)

        knots = torch.linspace(0., 1., M + 1, dtype=torch.float64)
        grid_cfg.set_density(knots, kappa.double().clamp_min(0.).pow(grid_cfg.gamma))
        return kappa

    def forward(
            self,
            ctx: Tensor,
            batch: Tensor,
            nu_batch: Tensor,
            neutrino_t: Optional[Tensor]=None,
            sampling: bool = True,
            noise: Optional[Tensor] = None,
            num_steps: Optional[Union[int, Tensor]] = None
        ) ->  Tensor:
        """
        If the module is in training state, `forward` returns diffusion loss;
        otherwise, returns the ODE results.
        :obj:`noise` and :obj:`num_steps` are passed to :meth:`solver`.
        """
        ctx_s = ctx

        loss = None
        if neutrino_t is not None:
            device = batch.device
            num_nu_per_event = degree(nu_batch)
            num_nu_per_event = num_nu_per_event[num_nu_per_event!=0]   # skip events with 0 neutrinos

            # Sample one timestep per event (with at least one neutrino)
            T = torch.rand(num_nu_per_event.size(0), device=device)
            T = T.repeat_interleave(num_nu_per_event.to(torch.int64))
            signal_rate, noise_rate, _ = self.scheduler(T)

            # ~N(0,1) or ~U(0,1) noise
            if self._use_gaussian_noise:
                N = torch.randn((ctx.size(0), self.d_target), device=device)
            else:
                N = torch.rand((ctx.size(0), self.d_target), device=device)
            # Diffused (noised) data
            D = neutrino_t * signal_rate + N * noise_rate

            # Denoising model
            noise_pred = self.Denoiser(D, ctx, T, nu_batch)

            loss = torch.nn.functional.mse_loss(noise_pred, N, reduction='none')
            loss = torch.sum(loss, dim=1)
            return (loss, self.solver(ctx_s, nu_batch, noise, num_steps)) if sampling else loss

        else:
            return self.solver(ctx_s, nu_batch, noise, num_steps)
