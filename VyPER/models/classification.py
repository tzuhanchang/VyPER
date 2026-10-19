import math
import torch

from torch import nn, Tensor
from torch.nn.functional import linear, dropout
from torch_geometric.utils import scatter
from typing import Optional

from .mlp import Mlp
from VyPER.utils import softmax


class ClassAttentionBlock(nn.Module):
    r"""Class-attention block: a single event token attends over the
    object tokens of its event and only the event token is updated.

    Tokens are not padded into a dense batch. As each event has a single
    query, the attention is computed per object token and normalised
    within each event, which supports ONNX export with data-dependent sizes.

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
        q = q + self._attention(self.norm_q(q), self.norm_kv(kv), index)
        q = q + self.mlp(self.norm_mlp(q))
        return q


class EventClassification(nn.Module):
    r"""Event classification head.

    Every node, edge and neutrino embedding of an event becomes a token
    (with a learned type embedding); each event has one edge token per
    directed edge. An event token, initialised from the
    global embedding :obj:`u`, attends over these tokens through a stack of
    :class:`ClassAttentionBlock` and is finally projected to class logits.

    Args:
        d_model (int): dimension of the node, edge, global and neutrino embeddings.
        d_out (int): number of classes.
        d_feedforward (int): token dimension and feed-forward hidden dimension.
        use_neutrino (bool): include neutrino context tokens.
        nheads (int, optional): number of attention heads. (default: 1)
        attn_depth (int, optional): number of class-attention blocks. (default: 1)
        dropout (float, optional): dropout probability. (default: 0.01)
        activation (callable, optional): feed-forward activation. (default: :obj:`nn.ReLU()`)
    """
    NODE, EDGE, NEUTRINO = 0, 1, 2

    def __init__(
            self,
            d_model: int,
            d_out: int,
            d_feedforward: int,
            use_neutrino: bool,
            nheads: int = 1,
            attn_depth: int = 1,
            dropout: float = 0.01,
            activation: callable = nn.ReLU()
        ) -> None:
        super().__init__()

        self._use_neutrino = use_neutrino

        # Token projections, one per object type, plus a learned type embedding
        self.node_proj = nn.Linear(d_model, d_feedforward)
        self.edge_proj = nn.Linear(d_model, d_feedforward)
        self.nu_proj   = nn.Linear(d_model, d_feedforward) if use_neutrino else None
        self.type_embed = nn.Embedding(3, d_feedforward)

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
                nu_ctx: Optional[Tensor] = None, nu_batch: Optional[Tensor] = None) -> Tensor:
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

        :rtype: :class:`Tensor` of shape [B, `d_out`]
        """
        # Build the object tokens
        tokens = [self.node_proj(x) + self.type_embed.weight[self.NODE],
                  self.edge_proj(edge_attr) + self.type_embed.weight[self.EDGE]]
        index  = [batch, batch[edge_index[0]]]
        if self._use_neutrino:
            tokens.append(self.nu_proj(nu_ctx) + self.type_embed.weight[self.NEUTRINO])
            index.append(nu_batch)
        tokens = torch.cat(tokens, dim=0)
        index  = torch.cat(index, dim=0)

        # Event token attends over the object tokens
        q = self.query_proj(u)
        for block in self.blocks:
            q = block(q, tokens, index)

        return self.out_proj(self.norm_out(q))
