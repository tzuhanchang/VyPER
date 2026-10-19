import torch
import math

from torch import nn
from torch.nn.functional import relu

from .mlp import Mlp
from VyPER.utils import softmax


class HyperedgeBlock(nn.Module):
    r"""The hyperedge block.

    Each node is embedded together with the global features of its event, and
    each candidate hyperedge (a set of :obj:`r` nodes) sums the embeddings of
    its nodes. The hyperedge embedding is concatenated with a weighted copy,
    whose weights are normalised over the hyperedges of the same event, and is
    then classified.

    Args:
        node_in_channels (int): number of node features of the input graph.
        node_out_channels (int): number of output features of each hyperedge,
            i.e. the number of hyperedge classes.
        global_in_channels (int): number of global features of the input graph.
        message_feats (int, optional): number of intermediate features, and of the
            latent hyperedge features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)
        return_latent (bool, optional): if set to :obj:`True`, also return a latent
            representation of each hyperedge, of size :obj:`message_feats`, e.g. used as
            hyperedge tokens by :class:`EventClassification`. (default :obj:`bool`=False)

    :rtype: :class:`Tuple[Tensor,Tensor]` of the hyperedge logits, of shape
        [N_h, `node_out_channels`], and the event of each hyperedge, of shape [N_h].
        If :obj:`return_latent` is :obj:`True`, :class:`Tuple[Tensor,Tensor,Tensor]`
        with the latent hyperedges, of shape [N_h, `message_feats`], in between.
    """
    def __init__(self, node_in_channels, node_out_channels, global_in_channels,
                 message_feats: int=32, dropout=0.01, return_latent: bool=False):
        super().__init__()
        self.node_in_channels = node_in_channels
        self.message_feats = message_feats
        self.return_latent = return_latent

        self.HyperedgeConstructor = Mlp(
            d_in=node_in_channels+global_in_channels,
            d_out=message_feats,
            d_hidden=message_feats,
            depth=3,
            activation=nn.ReLU(),
            dropout=dropout,
            bias=True
        )
        self.FinalHyperedgeLayer = Mlp(
            d_in=message_feats*2,
            d_out=node_out_channels,
            d_hidden=message_feats,
            depth=2,
            activation=nn.ReLU(),
            dropout=dropout,
            bias=True
        )
        self.weight = nn.Parameter(torch.empty((message_feats, message_feats)))

        if self.return_latent:
            self.LatentHyperedgeLayer = nn.Linear(message_feats*2, message_feats)
        else:
            self.register_parameter("LatentHyperedgeLayer", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.HyperedgeConstructor.reset_parameters()
        self.FinalHyperedgeLayer.reset_parameters()
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.return_latent:
            self.LatentHyperedgeLayer.reset_parameters()

    def __hyperedge_finding__(self, x, hyperedge_index, r):
        hyperedge_index = hyperedge_index.permute(dims=(1,0))
        x_hyper = torch.gather(
            x.unsqueeze(1).expand([-1,r,-1]),
            0,
            hyperedge_index.unsqueeze(2).expand([-1,-1,self.message_feats])
        ).transpose(1,2)
        return x_hyper.sum(2)

    def weighting(self, x_hyper, batch_hyper, num_events):
        coefficient = softmax(x_hyper, index=batch_hyper, dim_size=num_events)
        return coefficient * relu(torch.mm(x_hyper, self.weight), inplace=True)

    def forward(self, x, u, batch, hyperedge_index, batch_hyper, r):
        x_hyper = self.HyperedgeConstructor(torch.cat([x, u[batch]], dim=1).float())
        x_hyper = self.__hyperedge_finding__(x_hyper, hyperedge_index, r)
        x_hyper_hat = self.weighting(x_hyper, batch_hyper, u.size(0))
        out = torch.cat([x_hyper, x_hyper_hat], dim=1).float()
        if self.return_latent:
            return self.FinalHyperedgeLayer(out), self.LatentHyperedgeLayer(out), batch_hyper
        else:
            return self.FinalHyperedgeLayer(out), batch_hyper
