import math
import torch

from torch import nn, Tensor
from torch.nn.functional import linear, dropout
from torch_geometric.utils import scatter
from typing import Optional

from .mlp import Mlp
from .attention import split_last
from VyPER.utils import softmax


class ClassAttentionBlock(nn.Module):
    r"""Class-attention block: a single event token attends over the
    object tokens of its event and only the event token is updated.

    Tokens are not padded into a dense batch. As each event has a single
    query, the attention is computed per object token and normalised
    within each event, which supports ONNX export with data-dependent sizes.

    The normalised object tokens are conditioned on the event token: they are
    modulated by a shift and a scale predicted from the event token (as adaLN
    in DiT), so the keys and values depend on what the event token has gathered
    in the previous blocks. The modulation is initialised to zero, so a freshly
    initialised block attends over the unmodulated tokens.

    Args:
        d_embed (int): embedding dimension.
        d_feedforward (int): hidden dimension of the feed-forward MLP.
        nheads (int, optional): number of attention heads. (default: 1)
        dropout (float, optional): dropout probability. (default: 0.01)
        activation (callable, optional): feed-forward activation. (default: :obj:`nn.ReLU()`)
    """
    def __init__(
            self,
            d_embed: int,
            d_feedforward: int,
            nheads: int = 1,
            dropout: float = 0.01,
            activation: callable = nn.ReLU()
        ) -> None:
        super().__init__()

        self.norm_q  = nn.LayerNorm(d_embed)
        self.norm_kv = nn.LayerNorm(d_embed)
        self.attn = nn.MultiheadAttention(d_embed, nheads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_embed)
        self.mlp = Mlp(d_embed,
                       d_out=d_embed,
                       d_hidden=d_feedforward,
                       depth=0,
                       activation=activation,
                       dropout=dropout)

        # Conditioning of the object tokens on the event token, initialised to zero
        self.token_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_embed, 2 * d_embed))
        nn.init.zeros_(self.token_modulation[-1].weight)
        nn.init.zeros_(self.token_modulation[-1].bias)

        # Supported by `_attention`
        assert self.attn._qkv_same_embed_dim and self.attn.bias_k is None \
            and not self.attn.add_zero_attn, \
            "`ClassAttentionBlock` does not support `kdim`, `vdim`, `add_bias_kv` or `add_zero_attn`."

    def _attention(self, q: Tensor, kv: Tensor, index: Tensor) -> Tensor:
        r"""Attention of each event token over the object tokens of its event,
        with the parameters of :obj:`self.attn`.

        Args:
            q (torch.Tensor): event tokens of shape [B, `d_embed`].
            kv (torch.Tensor): object tokens of shape [T, `d_embed`].
            index (torch.Tensor): event of each object token, of shape [T].

        :rtype: :class:`torch.Tensor` of shape [B, `d_embed`]
        """
        mha = self.attn
        B, E = q.shape
        H = mha.num_heads
        w, b = mha.in_proj_weight, mha.in_proj_bias

        q = linear(q, w[:E], b[:E] if b is not None else None).reshape(B, H, E // H)
        k = linear(kv, w[E:2*E], b[E:2*E] if b is not None else None).reshape(-1, H, E // H)
        v = linear(kv, w[2*E:], b[2*E:] if b is not None else None).reshape(-1, H, E // H)

        # Scaled dot-product of each object token with the event token, [T, H]
        scores = (q.index_select(0, index) * k).sum(dim=-1) * (1.0 / math.sqrt(E // H))
        weights = softmax(scores, index, dim_size=B)
        if self.training and mha.dropout > 0.:
            weights = dropout(weights, p=mha.dropout)
        out = scatter(weights.unsqueeze(-1) * v, index, dim=0, dim_size=B, reduce='sum')
        return mha.out_proj(out.reshape(B, E))

    def forward(self, q: Tensor, kv: Tensor, index: Tensor) -> Tensor:
        r"""
        Args:
            q (torch.Tensor): event tokens of shape [B, `d_embed`].
            kv (torch.Tensor): object tokens of shape [T, `d_embed`].
            index (torch.Tensor): event of each object token, of shape [T].
        """
        shift, scale = split_last(self.token_modulation(q), 2)
        kv_norm = self.norm_kv(kv) * (1 + scale.index_select(0, index)) + shift.index_select(0, index)
        q = q + self._attention(self.norm_q(q), kv_norm, index)
        q = q + self.mlp(self.norm_mlp(q))
        return q


class EventClassification(nn.Module):
    r"""Event classification head.

    Every node, edge, neutrino and hyperedge embedding of an event becomes
    a token (with a learned type embedding); each event has one edge token
    per directed edge. An event token, initialised from the
    global embedding :obj:`u`, attends over these tokens, conditioned on the
    event token, through a stack of :class:`ClassAttentionBlock` and is finally
    projected to class logits.

    Args:
        d_model (int): dimension of the node, edge, global and neutrino embeddings.
        d_out (int): number of classes.
        d_feedforward (int): token dimension and feed-forward hidden dimension.
        use_neutrino (bool): include neutrino context tokens.
        use_hyperedge (bool, optional): include hyperedge tokens, from the latent
            hyperedges of :class:`HyperedgeBlock` (:obj:`return_latent=True`). (default: :obj:`False`)
        d_hyperedge (int, optional): dimension of the latent hyperedges. Required
            if :obj:`use_hyperedge` is :obj:`True`. (default: :obj:`None`)
        nheads (int, optional): number of attention heads. (default: 1)
        attn_depth (int, optional): number of class-attention blocks. (default: 1)
        dropout (float, optional): dropout probability. (default: 0.01)
        activation (callable, optional): feed-forward activation. (default: :obj:`nn.ReLU()`)
    """
    NODE, EDGE, NEUTRINO, HYPEREDGE = 0, 1, 2, 3

    def __init__(
            self,
            d_model: int,
            d_out: int,
            d_feedforward: int,
            use_neutrino: bool,
            use_hyperedge: bool = False,
            d_hyperedge: Optional[int] = None,
            nheads: int = 1,
            attn_depth: int = 1,
            dropout: float = 0.01,
            activation: callable = nn.ReLU()
        ) -> None:
        super().__init__()

        self._use_neutrino = use_neutrino
        self._use_hyperedge = use_hyperedge
        if use_hyperedge:
            assert d_hyperedge is not None, "`d_hyperedge` is required if `use_hyperedge` is True."

        # Token projections, one per object type, plus a learned type embedding
        self.node_proj = nn.Linear(d_model, d_feedforward)
        self.edge_proj = nn.Linear(d_model, d_feedforward)
        self.nu_proj   = nn.Linear(d_model, d_feedforward) if use_neutrino else None
        self.hyper_proj = nn.Linear(d_hyperedge, d_feedforward) if use_hyperedge else None
        self.type_embed = nn.Embedding(4 if use_hyperedge else 3, d_feedforward)

        # Event token, initialised from the global embedding
        self.query_proj = nn.Linear(d_model, d_feedforward)

        self.blocks = nn.ModuleList([
            ClassAttentionBlock(d_feedforward, d_feedforward, nheads=nheads,
                                dropout=dropout, activation=activation)
            for _ in range(attn_depth)])

        self.norm_out = nn.LayerNorm(d_feedforward)
        self.out_proj = Mlp(d_feedforward,
                            d_out=d_out,
                            d_hidden=d_feedforward,
                            depth=2,
                            activation=activation,
                            dropout=dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, u: Tensor, batch: Tensor,
                nu_ctx: Optional[Tensor] = None, nu_batch: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None, hyperedge_batch: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): node embeddings of shape [N, `d_model`].
            edge_index (torch.Tensor): edge indices of shape [2, E].
            edge_attr (torch.Tensor): edge embeddings of shape [E, `d_model`].
            u (torch.Tensor): global embeddings of shape [B, `d_model`].
            batch (torch.Tensor): node to event assignment of shape [N].
            nu_ctx (torch.Tensor, optional): neutrino context of shape [N_nu, `d_model`].
            nu_batch (torch.Tensor, optional): neutrino to event assignment of shape [N_nu].
                Events without neutrinos simply contribute no neutrino tokens.
            hyperedge_attr (torch.Tensor, optional): latent hyperedges of shape [N_h, `d_hyperedge`].
            hyperedge_batch (torch.Tensor, optional): hyperedge to event assignment of shape [N_h].

        :rtype: :class:`Tensor` of shape [B, `d_out`]
        """
        # Build the object tokens
        tokens = [self.node_proj(x) + self.type_embed.weight[self.NODE],
                  self.edge_proj(edge_attr) + self.type_embed.weight[self.EDGE]]
        index  = [batch, batch[edge_index[0]]]
        if self._use_neutrino:
            tokens.append(self.nu_proj(nu_ctx) + self.type_embed.weight[self.NEUTRINO])
            index.append(nu_batch)
        if self._use_hyperedge:
            tokens.append(self.hyper_proj(hyperedge_attr) + self.type_embed.weight[self.HYPEREDGE])
            index.append(hyperedge_batch)
        tokens = torch.cat(tokens, dim=0)
        index  = torch.cat(index, dim=0)

        # Event token attends over the object tokens
        q = self.query_proj(u)
        for block in self.blocks:
            q = block(q, tokens, index)

        return self.out_proj(self.norm_out(q))
