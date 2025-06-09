import torch

from torch import nn, Tensor
from torch_geometric.utils import scatter
from typing import Optional


def EdgeLoss(edge_attr_out: Tensor, edge_attr_t: Tensor, edge_attr_batch: Tensor,
             topo_max_num_edges: Optional[Tensor]=None, reduction: str='mean') -> Tensor:
    r"""Calculate per graph edge loss.

    Args:
        edge_attr_out (Tensor): output edge features.
        edge_attr_t (Tensor): edge targets.
        edge_attr_batch (Tensor): edge batch.
        topo_max_num_edges (optional: Tensor): maximun number of edges could 
            exist in a graph (default: :obj:`None`)
        reduction (optional: str): the reduce operation (default: 'mean').

    :rtype: :class:`Tensor`
    """
    l = nn.functional.cross_entropy(edge_attr_out, edge_attr_t.float(),reduction='none')
    if topo_max_num_edges is not None:
        weight = scatter(edge_attr_t[:,:-1], edge_attr_batch, reduce='sum').sum(1)/topo_max_num_edges.flatten()
        return scatter(l, edge_attr_batch, reduce=reduction) * weight
    return scatter(l, edge_attr_batch, reduce=reduction)


def HyperedgeLoss(x_out: Tensor, x_t: Tensor, x_t_batch: Tensor,
                  topo_max_num_hyperedges: Optional[Tensor]=None, reduction: str='mean') -> Tensor:
    r"""Calculate per graph hyperedge loss.

    Args:
        x_out (Tensor): output hyperedge features.
        x_t (Tensor): hyperedge targets.
        x_t_batch (Tensor): hyperedge batch.
        topo_max_num_hyperedges (optional: Tensor): maximun number of hyperedges could 
            exist in a graph (default: :obj:`None`)
        reduction (optional: str): the reduce operation (default: 'mean').

    :rtype: :class:`Tensor`
    """
    l = nn.functional.binary_cross_entropy(x_out, x_t.float(),reduction='none')
    if topo_max_num_hyperedges is not None:
        weight = scatter(x_t, x_t_batch, reduce='sum')/topo_max_num_hyperedges
        return scatter(l.flatten(), x_t_batch, reduce=reduction) * weight.flatten()
    return scatter(l.flatten(), x_t_batch, reduce=reduction)


def DiffusionLoss(nu_loss: Tensor, nu_batch: Tensor, reduction: str='mean') -> Tensor:
    r"""Calculate per graph diffusion loss.

    Args:
        nu_loss (Tensor): diffusion loss.
        nu_batch (Tensor): neutrino batch.
        reduction (optional: str): the reduce operation (default: 'mean').
    
    :rtype: :class:`Tensor`
    """
    return scatter(nu_loss.flatten(), nu_batch, reduce=reduction)


def CombinedLoss(edge_loss: Tensor, nu_loss: Tensor, hyperedge_loss: Optional[Tensor]=None,
                 alpha: float=0.5, eta: float=0.5, reduction='mean') -> Tensor:
    r"""Get combined loss.

    Args:
        egde_loss (Tensor): edge loss.
        nu_loss (Tensor): diffusion loss.
        hyperedge_loss (Tensor): hyperedge loss.
        alpha (optional: Tensor): hyperedge loss weight. (default: 0.5)
        eta (optional: Tensor): diffusion loss weight. (default: 0.5)
        reduction (optional: str): the reduce operation (default: 'mean').

    :rtype: :class:`Tensor`
    """
    if hyperedge_loss is not None:
        l = eta * nu_loss + (1-eta) * (alpha * hyperedge_loss + ((1-alpha) * edge_loss))
    else:
        l = eta * nu_loss + (1-eta) * edge_loss

    if reduction == 'mean':
        rd = torch.mean
    elif reduction == 'max':
        rd = torch.max
    elif reduction == 'min':
        rd = torch.min
    elif reduction == 'sum':
        rd = torch.sum
    else:
        NotImplementedError("Available reduction methods are: 'mean', 'max', 'min' and 'sum'.")
    return rd(l)
    