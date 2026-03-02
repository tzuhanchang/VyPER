import torch

from torch import nn
from typing import Optional

from .mlp import Mlp
from .message import EdgeModel, NodeModel, GlobalModel, NeutrinoModel


class MPBlock(nn.Module):
    r"""A message passing iteration.

    Args:
        edge_model (torch.nn.Module): graph transformer edge convolution.
        node_model (torch.nn.Module): graph transformer node convolution.
        gloabl_model (torch.nn.Module): graph transformer global convolution.
        neutrino_model (optional, torch.nn.Module): conditional neutrino message passing.

    :rtype: :class:`Tuple[torch.Tensor,torch.Tensor,torch.Tensor]
    """
    def __init__(
            self,
            edge_model: torch.nn.Module,
            node_model: torch.nn.Module,
            global_model: torch.nn.Module,
            neutrino_model: Optional[torch.nn.Module]=None
        ) -> None:
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.glob_model = global_model
        self.neutrino_model = neutrino_model

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for model in [self.edge_model, self.node_model, self.glob_model]:
            model.reset_parameters()
        if self.neutrino_model is not None:
            self.neutrino_model.reset_parameters()

    def forward(self, x, batch, edge_attr, edge_index, u, lep_node=None, lep_forward_edge=None):
        if self.neutrino_model is not None:
            if lep_node is None or lep_forward_edge is None:
                raise ValueError("Please provide structures used for neutrino message forwarding.")
            ctx, x = self.neutrino_model(x, edge_index, edge_attr, u, batch, lep_node, lep_forward_edge)

        edge_attr = self.edge_model(x, edge_index, edge_attr, u, batch)
        x = self.node_model(x, edge_index, edge_attr, u, batch)
        u = self.glob_model(x, edge_index, edge_attr, u, batch)
        return (x, edge_attr, u) if self.neutrino_model is None else (x, edge_attr, u, ctx)


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
            use_edge: bool = True,
            use_neutrino: bool = True
        ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self._use_edge = use_edge
        self._use_neutrino = use_neutrino

        for i in range(num_layers):
            if i == 0 and num_layers == 1:
                setattr(self, 'MessagePassing' + str(i),
                    MPBlock(
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
                            d_out=node_out_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        ),
                        GlobalModel(
                            d_node=node_out_channels,
                            d_edge=message_feats,
                            d_glob=global_in_channels,
                            d_out=global_out_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        ),
                        neutrino_model=NeutrinoModel(
                            d_node=node_in_channels,
                            d_edge=edge_in_channels,
                            d_glob=global_in_channels,
                            d_out=nu_ctx_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        ) if self._use_neutrino else None
                    )
                )
            elif i == 0 and num_layers > 1:
                setattr(self, 'MessagePassing' + str(i),
                    MPBlock(
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
                            d_edge=message_feats,
                            d_glob=global_in_channels,
                            d_out=message_feats,
                            d_embed=message_feats,
                            dropout=dropout
                        ),
                        neutrino_model=NeutrinoModel(
                            d_node=node_in_channels,
                            d_edge=edge_in_channels,
                            d_glob=global_in_channels,
                            d_out=nu_ctx_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        ) if self._use_neutrino else None
                    )
                )
            elif i == num_layers-1:
                setattr(self, 'MessagePassing' + str(i),
                    MPBlock(
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
                            d_edge=message_feats,
                            d_glob=message_feats,
                            d_out=global_out_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        ),
                        neutrino_model=NeutrinoModel(
                            d_node=message_feats,
                            d_edge=message_feats,
                            d_glob=message_feats,
                            d_out=nu_ctx_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        ) if self._use_neutrino else None
                    )
                )
            else:
                setattr(self, 'MessagePassing' + str(i),
                    MPBlock(
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
                            d_edge=message_feats,
                            d_glob=message_feats,
                            d_out=message_feats,
                            d_embed=message_feats,
                            dropout=dropout
                        ),
                        neutrino_model=NeutrinoModel(
                            d_node=message_feats,
                            d_edge=message_feats,
                            d_glob=message_feats,
                            d_out=nu_ctx_channels,
                            d_embed=message_feats,
                            dropout=dropout
                        ) if self._use_neutrino else None
                    )
                )

        if self._use_edge:
            self.FinalEdgeLayer = Mlp(
                d_in=self.num_layers*message_feats,
                d_out=edge_out_channels,
                d_hidden=message_feats,
                depth=2,
                activation=nn.ReLU(),
                dropout=dropout,
                bias=True)
        else:
            self.register_parameter('FinalEdgeLayer', None)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for i in range(self.num_layers):
            for layer in getattr(self, 'MessagePassing' + str(i)).children():
                layer.reset_parameters()
        if self._use_edge:
            self.FinalEdgeLayer.reset_parameters()

    def forward(self, x, edge_index, edge_attr, u, batch,
                lep_node: Optional[torch.tensor]=None,
                lep_forward_edge: Optional[torch.tensor]=None):
        if self._use_neutrino:
            assert lep_node is not None and lep_forward_edge is not None

        nu_ctx = []; edge_ctx = []
        # Message Passing Step
        for i in range(self.num_layers):
            if self._use_neutrino:
                x, edge_attr, u, ctx = getattr(self, 'MessagePassing' + str(i))(
                    x, batch, edge_attr, edge_index, u, lep_node, lep_forward_edge
                )
                nu_ctx.append(ctx)
            else:
                x, edge_attr, u = getattr(self, 'MessagePassing' + str(i))(
                    x, batch, edge_attr, edge_index, u
                )
            edge_ctx.append(edge_attr)

        # Summarising
        if self._use_edge:
            edge_attr = self.FinalEdgeLayer(torch.cat(edge_ctx, dim=1))
        nu_ctx = torch.cat(nu_ctx, dim=1).float() if self._use_neutrino else None
        return [x, edge_attr, u, nu_ctx]
