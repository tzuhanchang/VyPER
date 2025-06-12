import torch
import math

from torch import nn
from torch.nn.functional import relu

from .mlp import Mlp
from VyPER.utils import softmax


class HyperedgeBlock(nn.Module):
    r"""The hyperedge block.

    Args:
        n_node_feats (int): number of node features of input graph.
        n_node_feats_out (int): number of node features of the output graph.
        message_feats (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)

    :rtype: :class:`Tuple[Tensor,Tensor]`
    """
    def __init__(self, node_in_channels, node_out_channels, global_in_channels,
                 message_feats: int=32, dropout=0.01):
        super().__init__()
        self.node_in_channels = node_in_channels
        self.message_feats = message_feats

        self.hyperedge_constructor = Mlp(
            d_in=node_in_channels+global_in_channels,
            d_out=message_feats,
            d_hidden=message_feats,
            depth=3,
            activation=nn.ReLU(),
            dropout=dropout,
            bias=True
        )
        self.final_hyperedge_layer = Mlp(
            d_in=message_feats*2,
            d_out=node_out_channels,
            d_hidden=message_feats,
            depth=3,
            activation=nn.ReLU(),
            dropout=dropout,
            bias=True
        )
        self.weight = nn.Parameter(torch.empty((message_feats, message_feats)))

        self.reset_parameters()

    def reset_parameters(self):
        self.hyperedge_constructor.reset_parameters()
        self.final_hyperedge_layer.reset_parameters()
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def __hyperedge_finding__(self, x, hyperedge_index, r):
        hyperedge_index = hyperedge_index.permute(dims=(1,0))
        x_hyper = torch.gather(
            x.unsqueeze(1).expand([-1,r,-1]),
            0,
            hyperedge_index.unsqueeze(2).expand([-1,-1,self.message_feats])
        ).transpose(1,2)
        return x_hyper.sum(2)

    def weighting(self, x_hyper, batch_hyper):
        coefficient = softmax(x_hyper, index=batch_hyper, dim_size=x_hyper.size(0))
        return coefficient * relu(torch.mm(x_hyper, self.weight), inplace=True)

    def forward(self, x, u, batch, hyperedge_index, batch_hyper, r):
        x_hyper = self.hyperedge_constructor(torch.cat([x, u[batch]], dim=1).float())
        x_hyper = self.__hyperedge_finding__(x_hyper, hyperedge_index, r)
        x_hyper_hat = self.weighting(x_hyper, batch_hyper)
        out = torch.cat([x_hyper, x_hyper_hat], dim=1).float()
        return self.final_hyperedge_layer(out), batch_hyper