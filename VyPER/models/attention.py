import math
import torch

from torch import Tensor
from typing import Optional, Tuple, List, NamedTuple
from torch.nn import Module, Sequential as Seq, LayerNorm, Linear, Dropout, GELU, SiLU, MultiheadAttention, ModuleList
from torch.nn.functional import linear, pad, dropout


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def split_last(x: Tensor, n: int) -> List[Tensor]:
    r"""Split the last dimension of :obj:`x` into :obj:`n` equal chunks.
    Same as `x.chunk(n, dim=-1)`, but exported to ONNX as `Slice` instead
    of the slower `SplitToSequence`.

    Args:
        x (torch.Tensor): input tensor.
        n (int): number of chunks.

    :rtype: :class:`List[torch.Tensor]`
    """
    d = x.size(-1) // n
    return [x[..., i*d:(i+1)*d] for i in range(n)]


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

        # Supported by `_attention`
        assert self.attn._qkv_same_embed_dim and not self.attn.batch_first \
            and not self.attn.add_zero_attn, \
            "`DiTBlock` does not support `kdim`, `vdim`, `batch_first` or `add_zero_attn`."

    def _attention(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Self-attention with the parameters of :obj:`self.attn`, written out
        explicitly to support ONNX export with data-dependent input sizes.

        Args:
            x (torch.Tensor): input sequence of shape [L, B, `d_embed`].
            key_padding_mask (torch.Tensor, optional): boolean mask of shape [B, L],
                :obj:`True` marks padded positions. (default :obj:`None`)

        :rtype: :class:`torch.Tensor`
        """
        mha = self.attn
        L, B, E = x.shape
        H = mha.num_heads

        q, k, v = split_last(linear(x, mha.in_proj_weight, mha.in_proj_bias), 3)

        # Learned key/value bias (`add_bias_kv`)
        if mha.bias_k is not None:
            k = torch.cat([k, mha.bias_k.expand(1, B, E)], dim=0)
            v = torch.cat([v, mha.bias_v.expand(1, B, E)], dim=0)
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1), value=False)

        # [S, B, E] -> [B, H, S, E/H]
        q, k, v = (t.reshape(t.size(0), B, H, E // H).permute(1, 2, 0, 3) for t in (q, k, v))

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(E // H))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        if self.training and mha.dropout > 0.:
            weights = dropout(weights, p=mha.dropout)
        out = torch.matmul(weights, v)
        out = out.permute(2, 0, 1, 3).reshape(L, B, E)
        return mha.out_proj(out)

    def forward(self, x: Tensor, c: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): input sequence of shape [L, B, `d_embed`].
            c (torch.Tensor): conditioning vectors of shape [L, B, `d_embed`].
            key_padding_mask (torch.Tensor, optional): boolean mask of shape [B, L],
                :obj:`True` marks padded positions, which are ignored as attention keys.
                (default :obj:`None`)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = split_last(self.adaLN_modulation(c), 6)
        mod = modulate(self.norm1(x), shift_msa, scale_msa)
        attn = self._attention(mod, key_padding_mask)
        if key_padding_mask is not None:
            # Zero the attention output at padded positions. Not required for correctness
            # (padded positions are never used as keys and are dropped after unbatching),
            # but keeps padded rows from carrying spurious values between blocks.
            attn = attn * (~key_padding_mask).transpose(0,1).unsqueeze(-1)
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
        shift, scale = split_last(self.adaLN_modulation(c), 2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SamplingState(NamedTuple):
    r"""Tensors shared by all sampling steps, see :meth:`Denoiser.prepare_sampling`.

    Args:
        group (torch.Tensor): row of each neutrino in the dense layout.
        pos (torch.Tensor): column of each neutrino in the dense layout.
        key_padding_mask (torch.Tensor): boolean mask of shape [B, L],
            :obj:`True` marks padded positions.
        keep (torch.Tensor): mask of shape [B, L, 1], :obj:`0.` at padded positions.
        context_pre (torch.Tensor): timestep-independent part of the first
            `mlp_context` layer, of shape [B, L, `hidden_size`].
    """
    group: Tensor
    pos: Tensor
    key_padding_mask: Tensor
    keep: Tensor
    context_pre: Tensor


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
        self.d_ctx = d_ctx

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


    @staticmethod
    def _pack(nu_batch: Tensor) -> Tuple[Tensor, Tensor, int, int]:
        r"""Row and column of each neutrino in a dense [B, L] layout, with one
        row per event with neutrinos. An extra padded row is always added, as
        ONNX Runtime does not support empty batches.

        Args:
            nu_batch (torch.Tensor): sorted event index of each neutrino.

        :rtype: :class:`Tuple[torch.Tensor,torch.Tensor,int,int]`
        """
        # Consecutive event ids, skipping events without neutrinos
        _, group, counts = torch.unique_consecutive(nu_batch, return_inverse=True, return_counts=True)
        starts = torch.cumsum(counts, dim=0) - counts
        pos = torch.arange(nu_batch.size(0), device=nu_batch.device) - starts[group]

        B = counts.size(0) + 1
        L = torch.cat([counts, counts.new_ones(1)]).max().item()
        torch._check(L >= 1)
        return group, pos, B, L

    def forward(self, x: Tensor, c: Tensor, T: Tensor, nu_batch: Tensor):
        """
        :obj:`x` has a dimension of [N,`d_x`]
        :obj:`c` has a dimension of [N,`d_ctx`]
        :obj:`T` has a dimension of [N]
        :obj:`nu_batch` has a dimension of [N], sorted, mapping each neutrino to its event

        Note:
            :obj:`c` is the contaxt vector summarised from
            all message-passing layers.
            Events may have different numbers of neutrinos, including none.
        """
        # No neutrinos in the whole batch
        if x.size(0) == 0:
            return x.new_zeros((0, self.d_x))

        T = self.timestep_embedding(T)
        c = self.mlp_context(torch.cat([c,T],dim=1))
        x = self.mlp_noise(x)

        # Batch neutrinos by event, padded with zeros
        group, pos, B, L = self._pack(nu_batch)
        x_dense = x.new_zeros((B, L, x.size(1))).index_put((group, pos), x)
        c_dense = c.new_zeros((B, L, c.size(1))).index_put((group, pos), c)
        key_padding_mask = torch.ones((B, L), dtype=torch.bool, device=x.device) \
                                .index_put((group, pos), torch.zeros_like(group, dtype=torch.bool))

        # [B, L, d] -> [L, B, d]
        x_dense, c_dense = x_dense.transpose(0,1), c_dense.transpose(0,1)

        for block in self.blocks:
            x_dense = block(x_dense, c_dense, key_padding_mask)

        # Unbatch the output
        out = self.out(x_dense, c_dense).transpose(0,1)
        return out[group, pos]

    def prepare_sampling(self, c: Tensor, nu_batch: Tensor) -> SamplingState:
        r"""Compute the tensors shared by all sampling steps.

        Args:
            c (torch.Tensor): context vectors of shape [N, `d_ctx`].
            nu_batch (torch.Tensor): sorted event index of each neutrino, of shape [N].

        :rtype: :class:`SamplingState`
        """
        group, pos, B, L = self._pack(nu_batch)
        key_padding_mask = torch.ones((B, L), dtype=torch.bool, device=c.device) \
                                .index_put((group, pos), torch.zeros_like(group, dtype=torch.bool))
        keep = (~key_padding_mask).unsqueeze(-1).to(c.dtype)

        # Timestep-independent part of the first `mlp_context` layer
        lin = self.mlp_context[0]
        pre = linear(c, lin.weight[:, :self.d_ctx], lin.bias)
        context_pre = pre.new_zeros((B, L, pre.size(1))).index_put((group, pos), pre)
        return SamplingState(group, pos, key_padding_mask, keep, context_pre)

    def to_dense(self, x: Tensor, state: SamplingState) -> Tensor:
        r"""Convert a per-neutrino tensor of shape [N, d] to the dense layout [B, L, d].

        Args:
            x (torch.Tensor): per-neutrino tensor.
            state (SamplingState): from :meth:`prepare_sampling`.

        :rtype: :class:`torch.Tensor`
        """
        B, L = state.key_padding_mask.shape
        return x.new_zeros((B, L, x.size(1))).index_put((state.group, state.pos), x)

    def from_dense(self, x: Tensor, state: SamplingState) -> Tensor:
        r"""Convert a tensor in the dense layout [B, L, d] to a per-neutrino tensor of shape [N, d].

        Args:
            x (torch.Tensor): tensor in the dense layout.
            state (SamplingState): from :meth:`prepare_sampling`.

        :rtype: :class:`torch.Tensor`
        """
        return x[state.group, state.pos]

    def sample_step(self, x: Tensor, T: Tensor, state: SamplingState) -> Tensor:
        r"""Predict the noise at a timestep shared by all neutrinos. Same as
        :meth:`forward`, but in the dense layout and reusing :obj:`state`.

        Args:
            x (torch.Tensor): noised data of shape [B, L, `d_x`].
            T (torch.Tensor): timestep, a scalar tensor.
            state (SamplingState): from :meth:`prepare_sampling`.

        :rtype: :class:`torch.Tensor`
        """
        # Timestep-dependent part of the first `mlp_context` layer
        T_emb = self.timestep_embedding(T.reshape(1))
        lin = self.mlp_context[0]
        c = state.context_pre + linear(T_emb, lin.weight[:, self.d_ctx:])
        c = self.mlp_context[2](self.mlp_context[1](c))
        x = self.mlp_noise(x)

        # [B, L, d] -> [L, B, d]
        x, c = x.transpose(0,1), c.transpose(0,1)
        for block in self.blocks:
            x = block(x, c, state.key_padding_mask)
        return self.out(x, c).transpose(0,1)
