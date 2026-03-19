import torch

from torch.nn import Module, Sequential as Seq, Linear, ReLU, SiLU, LayerNorm
from torch_geometric.utils import scatter

from .mlp import Mlp


class EdgeModel(Module):
    r"""A callable which updates a graph's edge features based
    on its source and target node features, its current edge
    features and its global features.

    Args:
        d_node (int): number of node features of input graph.
        d_edge (int): number of edge features of input graph.
        d_glob (int): number of global features of input graph.
        d_out (int): number of edge features after updates.
        d_embed (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)
    """
    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_glob: int,
        d_out: int,
        d_embed: int=32,
        dropout: float=0.01
    ):
        super().__init__()

        self.edge_mlp = Mlp(d_in=2*d_node+d_edge+d_glob,
                            d_out=d_out,
                            d_hidden=d_embed,
                            depth=1,
                            dropout=dropout,
                            activation=ReLU())

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.edge_mlp.reset_parameters()

    def forward(self, x, edge_index, edge_attr, u, batch):
        src, dest = edge_index[0], edge_index[1]
        out = torch.cat([x[src], x[dest], edge_attr, u[batch[src]]], 1).float()
        return self.edge_mlp(out)


class NodeModel(Module):
    r"""A callable which updates a graph's node features based
    on its current node features, its graph connectivity, its edge
    features and its global features.

    Args:
        d_node (int): number of node features of input graph.
        d_edge (int): number of edge features of input graph.
        d_glob (int): number of global features of input graph.
        d_out (int): number of node features after updates.
        d_embed (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)
    """
    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_glob: int,
        d_out: int,
        d_embed: int=32,
        dropout: float=0.01
    ):
        super().__init__()

        self.msg_mlp  = Mlp(d_in=2*d_node+d_edge,
                            d_out=d_embed,
                            d_hidden=d_embed,
                            depth=1,
                            dropout=dropout,
                            activation=ReLU())
        self.node_mlp = Mlp(d_in=2*d_embed+d_node+d_glob,
                            d_out=d_out,
                            d_hidden=d_embed,
                            depth=1,
                            dropout=dropout,
                            activation=ReLU())

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.msg_mlp.reset_parameters()
        self.node_mlp.reset_parameters()

    def forward(self, x, edge_index, edge_attr, u, batch):
        src, dest = edge_index[0], edge_index[1]
        message = torch.cat([x[src], x[dest], edge_attr], 1).float()
        message = self.msg_mlp(message)
        agg_mean = scatter(message, dest, dim=0, dim_size=x.size(0), reduce='mean')
        agg_max = scatter(message, dest, dim=0, dim_size=x.size(0), reduce='max')
        out = torch.cat([x, agg_mean, agg_max, u[batch]], dim=1).float()
        return self.node_mlp(out) # update node with message


class GlobalModel(Module):
    r"""A callable which updates a graph's global features based
    on its node features, its graph connectivity, its edge features
    and its current global features.

    Args:
        d_node (int): number of node features of input graph.
        d_edge (int): number of edge features of input graph.
        d_glob (int): number of global features of input graph.
        d_out (int): number of global features after updates.
        d_embed (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)
    """
    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_glob: int,
        d_out: int,
        d_embed: int=32,
        dropout: float=0.01
    ):
        super().__init__()

        self.global_mlp = Mlp(d_in=2*d_node+d_glob,
                              d_out=d_out,
                              d_hidden=d_embed,
                              depth=1,
                              dropout=dropout,
                              activation=ReLU())

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.global_mlp.reset_parameters()

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([u,
            scatter(x, batch, dim=0, dim_size=u.size(0), reduce='mean'),
            scatter(x, batch, dim=0, dim_size=u.size(0), reduce='max')], dim=1).float()
        return self.global_mlp(out)


class NeutrinoModel(Module):
    r"""A callable which extracts neutrino context vectors based
    on its node features, its graph connectivity, its edge features
    and its current global features.

    Args:
        d_node (int): number of node features of input graph.
        d_edge (int): number of edge features of input graph.
        d_glob (int): number of global features of input graph.
        d_out (int): number of global features after updates.
        d_embed (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)
    """
    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_glob: int,
        d_out: int,
        d_embed: int=32,
        dropout: float=0.01
    ):
        super().__init__()

        self.ctx_lep_pass = Mlp(
            d_in=2*d_node+d_edge,
            d_out=d_embed,
            d_hidden=d_embed,
            depth=1,
            dropout=dropout,
            activation=SiLU()
        )
        self.ctx_nu_summarise = Mlp(
            d_in=d_node+d_glob+d_embed,
            d_out=d_embed,
            d_hidden=d_embed,
            depth=1,
            dropout=dropout,
            activation=SiLU()
        )
        self.ctx_nu_relative = Mlp(
            d_in=d_embed+d_glob,
            d_out=d_embed,
            d_hidden=d_embed,
            depth=1,
            dropout=dropout,
            activation=SiLU()
        )
        self.ctx_out = Mlp(
            d_in=2*d_embed,
            d_out=d_out,
            d_hidden=d_embed,
            depth=1,
            dropout=dropout,
            activation=SiLU()
        )
        self.backward_pass_mlp = Mlp(
            d_in=d_out,
            d_out=d_node,
            d_hidden=d_embed,
            depth=1,
            dropout=dropout,
            activation=SiLU()
        )
        self.backward_pass_norm = LayerNorm(d_node, elementwise_affine=False, eps=1e-6)
        self.backward_pass_modulation = Seq(SiLU(), Linear(d_node, 3 * d_node, bias=True))

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.ctx_lep_pass.reset_parameters()
        self.ctx_nu_summarise.reset_parameters()
        self.ctx_nu_relative.reset_parameters()
        self.ctx_out.reset_parameters()
        self.backward_pass_mlp.reset_parameters()
        self.backward_pass_norm.reset_parameters()
        for layer in self.backward_pass_modulation.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr, u, batch, lep_node, lep_forward_edge):
        num_neutrinos = scatter(lep_node, index=batch, dim=0, dim_size=u.size(0), reduce='sum')

        mask_n = lep_node.to(torch.bool)
        mask_e = lep_forward_edge.to(torch.bool)

        # Lepton forward messages
        fw_edges = edge_index[:,mask_e]
        src, dest = fw_edges[0], fw_edges[1]
        ctx = torch.cat([x[src], x[dest], edge_attr[mask_e]], 1).float()
        ctx = self.ctx_lep_pass(ctx) # message

        # Context vector - summarise information for a given neutrino
        nu_batch = torch.unique(batch)
        torch._check(u.shape[0] == nu_batch.shape[0])
        nu_batch = nu_batch.repeat_interleave(num_neutrinos)
        # nu_batch = torch.unique(batch).repeat_interleave(num_neutrinos)
        _, dest = torch.unique(dest, return_inverse=True)
        ctx = torch.cat([x[mask_n],
                         scatter(ctx, dest, dim=0, dim_size=(num_neutrinos.sum(0)).item(), reduce='sum'),
                         u[nu_batch]], dim=1).float()
        ctx = self.ctx_nu_summarise(ctx)

        # Relative context vector - summarise context vectors in a given event
        ctx_rel  = torch.cat([scatter(ctx, nu_batch, dim=0, dim_size=u.size(0), reduce='sum'), u], dim=1).float()
        ctx_rel  = self.ctx_nu_relative(ctx_rel)

        ctx_out  = self.ctx_out(torch.cat([ctx, ctx_rel[nu_batch]], dim=1))

        # Conditional AdaLN modulation for backward message passing
        shift, scale, scale_f  = self.backward_pass_modulation(x[mask_n]).chunk(3, dim=1)
        backward_pass = self.backward_pass_norm(self.backward_pass_mlp(ctx_out)) * (1 + scale) + shift

        # Backward message passing
        backward_shift, backward_scale = (torch.zeros_like(x, device=x.device, dtype=x.dtype),
                                          torch.zeros_like(x, device=x.device, dtype=x.dtype))
        backward_shift.index_copy(0, mask_n.nonzero(as_tuple=False).squeeze(1), backward_pass)
        backward_scale.index_copy(0, mask_n.nonzero(as_tuple=False).squeeze(1), scale_f)
        return ctx_out, (x * (1 + backward_scale) + backward_shift)