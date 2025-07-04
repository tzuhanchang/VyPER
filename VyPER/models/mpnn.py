import torch

from torch import nn
from torch_geometric.nn import MetaLayer
from typing import Optional

from .mlp import Mlp
from .message import EdgeModel, NodeModel, GlobalModel, NeutrinoModel


class MPNNs(nn.Module):
    r""" The Message Passing Neural Networks.

    Args:
        node_in_channels (int): number of node features of input graph.
        edge_in_channels (int): number of edge features of input graph.
        global_in_channels (int): number of global features of input graph.
        node_out_channels (int, optional): number of node features of output graph.
        edge_out_channels (int, optional): number of edge features of output graph.
        global_out_channels (int, optional): number of global features of output graph.
        message_feats (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)
        activation(callable, optional): activation function to apply. (default :obj:`callable`=torch.nn.Sigmoid)
        p_out(str, optional): the object which the `activation` is applied on. (default :obj:`None`)

    :rtype: :class:`Tuple[torch.Tensor,torch.Tensor]
    """
    def __init__(
            self,
            node_in_channels,
            edge_in_channels,
            global_in_channels,
            node_out_channels: int = 1,
            edge_out_channels: int = 1,
            global_out_channels: int = 1,
            num_layers: int = 1,
            nu_ctx_channels: int = 1,
            message_feats: int = 32,
            dropout: float = 0.01,
            use_neutrino: bool = True
        ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self._use_neutrino = use_neutrino

        for i in range(num_layers):
            if i == 0 and num_layers == 1:
                setattr(self, 'MessagePassing' + str(i),
                    MetaLayer(
                        EdgeModel(
                            d_node=node_in_channels,
                            d_edge=edge_in_channels,
                            d_glob=global_in_channels,
                            d_out=edge_out_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        ),
                        NodeModel(
                            d_node=node_in_channels,
                            d_edge=edge_out_channels,
                            d_glob=global_in_channels,
                            d_out=node_out_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        ),
                        GlobalModel(
                            d_node=node_out_channels,
                            d_glob=global_in_channels,
                            d_out=global_out_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        )
                    )
                )
            elif i == 0 and num_layers > 1:
                setattr(self, 'MessagePassing' + str(i),
                    MetaLayer(
                        EdgeModel(
                            d_node=node_in_channels,
                            d_edge=edge_in_channels,
                            d_glob=global_in_channels,
                            d_out=message_feats,
                            d_embed=message_feats,
                            dropout=dropout
                        ),
                        NodeModel(
                            d_node=node_in_channels,
                            d_edge=message_feats,
                            d_glob=global_in_channels,
                            d_out=message_feats,
                            d_embed=message_feats,
                            dropout=dropout
                        ),
                        GlobalModel(
                            d_node=message_feats,
                            d_glob=global_in_channels,
                            d_out=message_feats,
                            d_embed=message_feats,
                            dropout=dropout
                        )
                    )
                )
            elif i == num_layers-1:
                setattr(self, 'MessagePassing' + str(i),
                    MetaLayer(
                        EdgeModel(
                            d_node=message_feats,
                            d_edge=message_feats,
                            d_glob=message_feats,
                            d_out=message_feats,
                            d_embed=message_feats,
                            dropout=dropout
                        ),
                        NodeModel(
                            d_node=message_feats,
                            d_edge=message_feats,
                            d_glob=message_feats,
                            d_out=node_out_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        ),
                        GlobalModel(
                            d_node=node_out_channels,
                            d_glob=message_feats,
                            d_out=global_out_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        )
                    )
                )
            else:
                setattr(self, 'MessagePassing' + str(i),
                    MetaLayer(
                        EdgeModel(
                            d_node=message_feats,
                            d_edge=message_feats,
                            d_glob=message_feats,
                            d_out=message_feats,
                            d_embed=message_feats,
                            dropout=dropout
                        ),
                        NodeModel(
                            d_node=message_feats,
                            d_edge=message_feats,
                            d_glob=message_feats,
                            d_out=message_feats,
                            d_embed=message_feats,
                            dropout=dropout
                        ),
                        GlobalModel(
                            d_node=message_feats,
                            d_glob=message_feats,
                            d_out=message_feats,
                            d_embed=message_feats,
                            dropout=dropout
                        )
                    )
                )

            if self._use_neutrino:
                if i == 0:
                    setattr(self, 'NeutrinoModel' + str(i),
                        NeutrinoModel(
                            d_node=node_in_channels,
                            d_edge=edge_in_channels,
                            d_glob=global_in_channels,
                            d_out=nu_ctx_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        )
                    )
                elif i == num_layers-1:
                    setattr(self, 'NeutrinoModel' + str(i),
                        NeutrinoModel(
                            d_node=node_out_channels,
                            d_edge=message_feats,
                            d_glob=global_out_channels,
                            d_out=nu_ctx_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        )
                    )
                else:
                    setattr(self, 'NeutrinoModel' + str(i),
                        NeutrinoModel(
                            d_node=message_feats,
                            d_edge=message_feats,
                            d_glob=message_feats,
                            d_out=nu_ctx_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        )
                    )
        
        self.final_edge_layer = Mlp(
            d_in=message_feats,
            d_out=edge_out_channels,
            d_hidden=message_feats,
            depth=3,
            activation=nn.ReLU(),
            dropout=dropout,
            bias=True
        )


    def reset_parameters(self):
        # Reset all trainable parameters.
        for i in range(self.num_layers):
            for layer in getattr(self, 'MessagePassing' + str(i)).children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            if self._use_neutrino:
                for layer in getattr(self, 'NeutrinoModel' + str(i)).children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            self.final_edge_layer.reset_parameters()


    def forward(self, x, edge_index, edge_attr, u, batch,
                lep_node: Optional[torch.tensor]=None,
                lep_forward_edge: Optional[torch.tensor]=None):
        if self._use_neutrino:
            assert lep_node is not None and lep_forward_edge is not None
        nu_ctx = []
        # Message Passing Step
        for i in range(self.num_layers):
            if i == 0:
                if self._use_neutrino:
                    ctx, x_prime = getattr(self, 'NeutrinoModel' + str(i))(
                        x, edge_index, edge_attr, u, batch, lep_node, lep_forward_edge
                    )
                else:
                    x_prime = x
                x_prime, edge_attr_prime, u_prime = getattr(self, 'MessagePassing' + str(i))(
                    x_prime, edge_index, edge_attr, u, batch
                )
            else:
                if self._use_neutrino:
                    ctx, x_prime = getattr(self, 'NeutrinoModel' + str(i))(
                        x_prime, edge_index, edge_attr_prime, u_prime, batch, lep_node, lep_forward_edge
                    )
                    nu_ctx.append(ctx)
                x_prime, edge_attr_prime, u_prime = getattr(self, 'MessagePassing' + str(i))(
                    x_prime, edge_index, edge_attr_prime, u_prime, batch
                )

        # Summarising
        edge_attr_prime = self.final_edge_layer(edge_attr_prime)
        if self._use_neutrino:
            nu_ctx = torch.cat(nu_ctx, dim=1).float()
            return x_prime, edge_attr_prime, u_prime, nu_ctx
        else:
            return x_prime, edge_attr_prime, u_prime