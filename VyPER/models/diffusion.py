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
        num_sampling_steps: int = 1000
    ) -> None:
        super().__init__()
        self.d_ctx = d_ctx
        self.d_embed = d_embed
        self.d_target = d_target
        self.num_message_steps  = num_message_steps
        self.num_sampling_steps = num_sampling_steps
        self._use_gaussian_noise = (noise_distribution == "gaussian")
        self.scheduler = Scheduler()

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
        r"""Sample neutrinos from noise with deterministic (DDIM) sampling.

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

        # Tensors shared by all steps
        state = self.Denoiser.prepare_sampling(ctx, nu_batch)

        # Starting from pure noise and removing noise iteratively:
        D = self.Denoiser.to_dense(noise, state)
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
                T, D = self._step(T, dT, D, state)
                return i + 1, T, D

            i = torch.zeros((), dtype=torch.int64, device=ctx.device)
            _, _, D = while_loop(cond_fn, body_fn, (i, T, D))
        else:
            dT = 1 / num_steps
            for _ in range(num_steps):
                T, D = self._step(T, dT, D, state)

        return self.Denoiser.from_dense(D, state)

    def _step(self, T: Tensor, dT: Union[float, Tensor], D: Tensor, state: SamplingState) -> Tuple[Tensor, Tensor]:
        r"""One sampling step from :obj:`T` to :obj:`T - dT`.

        Args:
            T (torch.Tensor): current timestep, a scalar tensor.
            dT (float or torch.Tensor): timestep size.
            D (torch.Tensor): noised data in the dense layout, of shape [B, L, `d_target`].
            state (SamplingState): from :meth:`Denoiser.prepare_sampling`.

        :rtype: :class:`Tuple[torch.Tensor,torch.Tensor]`
        """
        # Diffusion Scheduler
        signal_rate_i, noise_rate_i, _ = self.scheduler(T)
        signal_rate_j, noise_rate_j, _ = self.scheduler(T-dT)
        # Denoising model
        noise_pred = self.Denoiser.sample_step(D, T, state)
        data_pred  = (D - noise_rate_i * noise_pred) / signal_rate_i

        D = signal_rate_j * data_pred + noise_rate_j * noise_pred
        # Reset padded positions
        return T - dT, D * state.keep

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
