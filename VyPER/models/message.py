import torch

from torch.nn import Module, Sequential as Seq, Linear, ReLU, Dropout, SiLU, LayerNorm
from torch_geometric.utils import scatter


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

        self.edge_mlp = Seq(
            Linear(2*d_node+d_edge+d_glob, d_embed),
            ReLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_embed),
            ReLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_out)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1).float()
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

        self.node_mlp_1 = Seq(
            Linear(2*d_node+d_edge, d_embed),
            ReLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_embed),
            ReLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_embed)
        )
        self.node_mlp_2 = Seq(
            Linear(2*d_embed+d_node+d_glob, d_embed),
            ReLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_embed),
            ReLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_out)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], x[col], edge_attr], 1).float()
        out = self.node_mlp_1(out) # message
        agg_mean = scatter(out, col, dim=0, dim_size=x.size(0), reduce='mean')
        agg_max = scatter(out, col, dim=0, dim_size=x.size(0), reduce='max')
        out = torch.cat([x, agg_mean, agg_max, u[batch]], dim=1).float()
        return self.node_mlp_2(out) # update node with message


class GlobalModel(Module):
    r"""A callable which updates a graph's global features based
    on its node features, its graph connectivity, its edge features
    and its current global features.

    Args:
        d_node (int): number of node features of input graph.
        d_glob (int): number of global features of input graph.
        d_out (int): number of global features after updates.
        d_embed (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)
    """
    def __init__(
        self,
        d_node: int,
        d_glob: int,
        d_out: int,
        d_embed: int=32,
        dropout: float=0.01
    ):
        super().__init__()

        self.global_mlp = Seq(
            Linear(2*d_node+d_glob, d_embed),
            ReLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_embed),
            ReLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_out)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u,
            scatter(x, batch, dim=0, dim_size=u.size(0), reduce='mean'),
            scatter(x, batch, dim=0, dim_size=u.size(0), reduce='max')], dim=1
        ).float()
        return self.global_mlp(out)


class NeutrinoModel(Module):
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

        self.ctx_lep_pass = Seq(
            Linear(2*d_node+d_edge, d_embed),
            SiLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_embed),
            SiLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_embed)
        )
        self.ctx_nu_summarise = Seq(
            Linear(d_embed+d_node+d_glob, d_embed),
            SiLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_embed),
            SiLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_embed)
        )
        self.ctx_nu_relative = Seq(
            Linear(d_embed+d_glob, d_embed),
            SiLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_embed),
            SiLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_embed)
        )
        self.ctx_out = Seq(
            Linear(2*d_embed, d_embed),
            SiLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_embed),
            SiLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_out)
        )

        self.backward_pass_mlp = Seq(
            Linear(d_out, d_embed),
            SiLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_embed),
            SiLU(),
            Dropout(p=dropout),
            Linear(d_embed, d_node)
        )
        self.backward_pass_norm = LayerNorm(d_node, elementwise_affine=False, eps=1e-6)
        self.backward_pass_modulation = Seq(
            SiLU(),
            Linear(d_node, 2 * d_node, bias=True)
        )

    def forward(self, x, edge_index, edge_attr, u, batch, lep_node, lep_forward_edge):
        num_neutrinos = scatter(lep_node, index=batch, dim=0, dim_size=u.size(0), reduce='sum')

        # Lepton forward messages
        src, dest = edge_index[:,lep_forward_edge.to(torch.bool)]
        ctx = torch.cat([x[src], x[dest], edge_attr[lep_forward_edge.to(torch.bool)]], 1).float()
        ctx = self.ctx_lep_pass(ctx) # message

        # Context vector - summarise information for a given neutrino
        nu_batch = torch.unique(batch).repeat_interleave(num_neutrinos)
        _, dest = torch.unique(dest, return_inverse=True)
        ctx = torch.cat([x[lep_node.to(torch.bool)], scatter(ctx, dest, dim=0, dim_size=num_neutrinos.sum(0), reduce='sum'), u[nu_batch]], dim=1).float()
        ctx = self.ctx_nu_summarise(ctx)

        # Relative context vector - summarise context vectors in a given event
        ctx_rel  = torch.cat([scatter(ctx, nu_batch, dim=0, dim_size=u.size(0), reduce='sum'), u], dim=1).float()
        ctx_rel  = self.ctx_nu_relative(ctx_rel)

        ctx_out  = self.ctx_out(torch.cat([ctx, ctx_rel[nu_batch]], dim=1))

        # Backward lepton pass
        backward_in = self.backward_pass_mlp(ctx_out)
        shift, scale = self.backward_pass_modulation(backward_in).chunk(2, dim=1)
        backward_pass = self.backward_pass_norm(backward_in) * (1 + scale) + shift
        backward_out = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        backward_out[lep_node.to(torch.bool)] = backward_pass
        return ctx_out, (x + backward_out)