import math
import torch

from torch import Tensor, nn
from torch_geometric.utils import degree
from typing import Optional

from .attention import Denoiser


class Scheduler(object):
    def __init__(self, SR_max: float=1., SR_min: float=1e-1, scheme: str='cosine'):
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

        return torch.cos(alpha), torch.sin(alpha), beta


class NeutrinoDiffusion(nn.Module):
    def __init__(
        self,
        d_ctx: int,
        d_embed: int,
        d_target: int,
        num_heads: int,
        num_message_steps: int,
        num_sampling_steps: int = 1000
    ) -> None:
        super().__init__()
        self.d_ctx = d_ctx
        self.d_embed = d_embed
        self.d_target = d_target
        self.num_message_steps  = num_message_steps
        self.num_sampling_steps = num_sampling_steps
        self.scheduler = Scheduler()

        self.Denoiser = Denoiser(
            d_embed=d_embed,
            depth=4,
            num_heads=num_heads,
            d_x=3,
            d_ctx=(d_ctx*num_message_steps),
            mlp_ratio=4.,
            dropout=0.
        )

    @torch.no_grad()
    def solver(self, ctx: Tensor, nu_batch: Tensor):
        device = ctx.device

        # ~N(0,1) noise
        N = torch.rand((ctx.size(0), self.d_target), device=device)
        # Timesteps
        T = torch.tensor([1.], device=device).expand(ctx.size(0))
        dT = 1 / self.num_sampling_steps

        # Starting from pure noise and removing noise iteratively:
        D = N
        for _ in range(self.num_sampling_steps):
            # Denoising model
            noise_pred = self.Denoiser(D, ctx, T, nu_batch)

            _, noise_rate, beta = self.scheduler(T)
            s = - noise_pred / noise_rate.view(-1,1)

            D += 0.5 * beta.view(-1,1) * (D + s) * dT
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

            # ~N(0,1) noise
            N = torch.rand((ctx.size(0), self.d_target), device=device)
            # Diffused (noised) data
            D = neutrino_t * signal_rate.view(-1,1) + N * noise_rate.view(-1,1)

            # Denoising model
            noise_pred = self.Denoiser(D, ctx, T, nu_batch)

            loss = torch.nn.functional.mse_loss(noise_pred, N, reduction='none')
            loss = torch.sum(loss, dim=1)
            return (loss, self.solver(ctx_s, nu_batch)) if sampling else loss

        else:
            return self.solver(ctx_s, nu_batch)