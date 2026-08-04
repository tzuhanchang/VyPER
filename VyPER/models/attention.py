import math
import torch

from torch import Tensor
from torch.nn import Module, Sequential as Seq, LayerNorm, Linear, Dropout, GELU, SiLU, MultiheadAttention, ModuleList

from VyPER.utils import group_batch


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DiTBlock(Module):
    r"""A custom DiT (https://github.com/facebookresearch/DiT) model.

    Args:
        d_embed (int): dimension of the embedding.
        numb_heads (int): number of attention heads.
        mlp_ratio (float, optional): MLP hidden layer dimension to
            embedding dimension ratio. (default :obj:`int`=4)
        dropout (float, optional): dropout fraction.
            (default :obj:`float`=0.)
        **block_kwargs: other arguments to the `torch.nn.MultiheadAttention`.
    """
    def __init__(
        self,
        d_embed: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        dropout: float = 0.,
        **block_kwargs
    ):
        super().__init__()
        self.d_embed = d_embed

        self.norm1 = LayerNorm(d_embed, elementwise_affine=False, eps=1e-6)
        self.attn  = MultiheadAttention(
            embed_dim=d_embed,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=True,
            **block_kwargs
        )
        self.norm2 = LayerNorm(d_embed, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(d_embed * mlp_ratio)
        self.mlp = Seq(
            Linear(d_embed, mlp_hidden_dim),
            GELU(approximate="tanh"),
            Dropout(dropout),
            Linear(mlp_hidden_dim, d_embed)
        )
        self.adaLN_modulation = Seq(
            SiLU(),
            Linear(d_embed, 6 * d_embed, bias=True)
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        mod = modulate(self.norm1(x), shift_msa, scale_msa)
        attn = self.attn(mod, mod, mod, need_weights=False)[0]
        x = x + gate_msa * attn
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTOut(Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = Seq(
            SiLU(),
            Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Denoiser(Module):
    r"""Core VyPER module, used to predict the gaussian noise at a
    given timestep using the context embedding extracted from the
    message-passing layers.

    Args:
        d_embed (int): dimension of the embedding.
        depth (int): number of `DiTBlock` used for denoising.
        num_heads (int): number of attention heads.
        d_x (int): dimension of the noised data.
        d_ctx (int): dimension of the context embedding.
        mlp_ratio (float, optional): MLP hidden layer dimension to
            embedding dimension ratio. (default :obj:`int`=4)
        dropout (float, optional): dropout fraction.
            (default :obj:`float`=0.)
    """
    def __init__(
            self,
            d_embed: int,
            depth: int,
            num_heads: int,
            d_x: int,
            d_ctx: int,
            mlp_ratio: float=4.0,
            dropout: float=0.
        ) -> None:
        super().__init__()

        self.d_embed = d_embed
        self.d_x = d_x

        self.hidden_size = int(mlp_ratio * d_embed)

        self.mlp_timestep = Seq(
            Linear(d_embed, d_embed),
            SiLU(),
            Linear(d_embed, d_embed))

        self.mlp_context = Seq(
            Linear(d_ctx+d_embed, self.hidden_size),
            SiLU(),
            Linear(self.hidden_size, d_embed))

        self.mlp_noise = Seq(
            Linear(d_x, self.hidden_size),
            SiLU(),
            Linear(self.hidden_size, d_embed))

        self.blocks = ModuleList([
            DiTBlock(
                d_embed,
                num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(depth)])

        self.out = DiTOut(d_embed, d_x)
    
    def timestep_embedding(
        self,
        timesteps: Tensor,
        max_period: int=10000
    ) -> Tensor:
        r"""Create sinusoidal timestep embeddings.
        This function is modified based on `glide-text2im` from OpenAI
        https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        Args:
            timesteps (torch.Tensor): a 1-D Tensor of N indices, one per batch element.
            dim (int): the dimension of the output.
            max_period (int, optional): controls the minimum frequency of the embeddings. (default :obj:`10000`)

        :rtype:`torch.Tensor`
        """
        half = self.d_embed // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_embed % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp_timestep(embedding)


    def forward(self, x: Tensor, c: Tensor, T: Tensor, nu_batch: Tensor):
        """
        :obj:`x` has a dimension of [N,`d_x`]
        :obj:`c` has a dimension of [N,`d_ctx`]
        :obj:`T` has a dimension of [N]

        Note:
            :obj:`c` is the contaxt vector summarised from
            all message-passing layers.
        """
        T = self.timestep_embedding(T)
        c = self.mlp_context(torch.cat([c,T],dim=1))
        x = self.mlp_noise(x)

        # `group_batch` adding a batch dimension
        x, mask = group_batch(x, nu_batch, return_mask=True)
        x = x.transpose(0,1)
        c = group_batch(c, nu_batch, return_mask=False).transpose(0,1)

        for block in self.blocks:
            x = block(x, c)

        # Unbatch the output
        x = self.out(x, c)
        x = x.transpose(0,1)[mask.bool()]
        return x