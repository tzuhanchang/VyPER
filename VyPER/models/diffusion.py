import math
import torch

from torch import Tensor, nn
from torch_geometric.utils import degree
from typing import Optional, Literal

from .attention import Denoiser


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

    @torch.no_grad()
    def solver(self, ctx: Tensor, nu_batch: Tensor):
        device = ctx.device

        # ~N(0,1) or ~U(0,1) noise
        if self._use_gaussian_noise:
            N = torch.randn((ctx.size(0), self.d_target), device=device)
        else:
            N = torch.rand((ctx.size(0), self.d_target), device=device)
        # Timesteps
        T = torch.ones((ctx.size(0),), dtype=ctx.dtype, device=device)
        dT = 1 / self.num_sampling_steps

        # Starting from pure noise and removing noise iteratively:
        D = N
        for _ in range(self.num_sampling_steps):
            # Diffusion Scheduler
            signal_rate_i, noise_rate_i, _ = self.scheduler(T)
            signal_rate_j, noise_rate_j, _ = self.scheduler(T-dT)
            # Denoising model
            noise_pred = self.Denoiser(D, ctx, T, nu_batch)
            data_pred  = (D - noise_rate_i * noise_pred) / signal_rate_i

            D = signal_rate_j * data_pred + noise_rate_j * noise_pred
            T = T - dT
        return D

    def forward(
            self,
            ctx: Tensor,
            batch: Tensor,
            nu_batch: Tensor,
            neutrino_t: Optional[Tensor]=None,
            sampling: bool = True
        ) ->  Tensor:
        """
        If the module is in training state, `forward` returns diffusion loss;
        otherwise, returns the ODE results.
        """
        ctx_s = ctx

        loss = None
        if neutrino_t is not None:
            device = batch.device
            num_graphs = len(degree(batch))

            # Sample timesteps
            T = torch.rand(num_graphs, device=device)
            T = T.repeat_interleave(degree(nu_batch).to(torch.int64))
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
            return (loss, self.solver(ctx_s, nu_batch)) if sampling else loss

        else:
            return self.solver(ctx_s, nu_batch)
