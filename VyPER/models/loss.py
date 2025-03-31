import torch

from torch import nn, Tensor
from torch_geometric.utils import scatter
from torch_hep.lorentz import MomentumTensor
from typing import Optional


def DeltaR_DiffusionLoss(input: Tensor, target: Tensor) -> Tensor:
    r"""Calculate angular distance loss between :obj:`input`
    and :obj:`Tensor`

    Args:
        input (Tensor): Predict noise in normalised (px,py,pz) space.
        target (Tensor): target noise in normalised (px,py,pz) space.

    :rtype: :class:`Tensor`
    """
    assert input.size(1)==3 and target.size(1)==3

    input_e  = ((input[:,0]*input[:,0]) + (input[:,1]*input[:,1])
              + (input[:,2]*input[:,2])).view(-1,1)
    target_e = ((target[:,0]*target[:,0]) + (target[:,1]*target[:,1])
              + (target[:,2]*target[:,2])).view(-1,1)

    input_p4  = MomentumTensor(torch.cat([input_e, input], dim=1))
    target_p4 = MomentumTensor(torch.cat([target_e, target], dim=1))

    dR = torch.sqrt((target_p4.eta - input_p4.eta) * (target_p4.eta - input_p4.eta)
                  + (target_p4.phi - input_p4.phi) * (target_p4.phi - input_p4.phi))
    return dR.view(-1,1)


def EdgeLoss(edge_attr_out: Tensor, edge_attr_t: Tensor, edge_attr_batch: Tensor,
             reduction: str='mean') -> Tensor:
    r"""Calculate per graph edge loss.

    Args:
        edge_attr_out (Tensor): output edge features.
        edge_attr_t (Tensor): edge targets.
        edge_attr_batch (Tensor): edge batch.
        reduction (optional: str): the reduce operation (default: 'mean').

    :rtype: :class:`Tensor`
    """
    l = nn.functional.binary_cross_entropy(edge_attr_out, edge_attr_t.float(),reduction='none')
    return scatter(l, edge_attr_batch, reduce=reduction)


def HyperedgeLoss(x_out: Tensor, x_t: Tensor, x_t_batch: Tensor,
                  reduction: str='mean') -> Tensor:
    r"""Calculate per graph hyperedge loss.

    Args:
        x_out (Tensor): output hyperedge features.
        x_t (Tensor): hyperedge targets.
        x_t_batch (Tensor): hyperedge batch.
        reduction (optional: str): the reduce operation (default: 'mean').

    :rtype: :class:`Tensor`
    """
    l = nn.functional.binary_cross_entropy(x_out, x_t.float(),reduction='none')
    l = scatter(l.flatten(), x_t_batch, reduce=reduction)
    # Mask for events with no target hyperedges
    loss_masks = scatter(x_t, x_t_batch, reduce='sum') > 0
    return l, loss_masks


def DiffusionLoss(nu_loss: Tensor, nu_batch: Tensor, reduction: str='mean') -> Tensor:
    r"""Calculate per graph diffusion loss.

    Args:
        nu_loss (Tensor): diffusion loss.
        nu_batch (Tensor): neutrino batch.
        reduction (optional: str): the reduce operation (default: 'mean').
    
    :rtype: :class:`Tensor`
    """
    return scatter(nu_loss.flatten(), nu_batch, reduce=reduction)


def CombinedLoss(edge_loss: Tensor, nu_loss: Tensor, #hyperedge_loss: Tensor
                 hyperedge_loss_mask: Optional[Tensor]=None,
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
    # if hyperedge_loss_mask is not None:
         # Ignore `loss_hyperedge` for an event that has no labelled hyperedges:
        # with torch.no_grad():
        #     device = loss_hyperedge.device
        #     loss_shape = loss_hyperedge.shape

        #     alpha = torch.full(loss_shape, alpha, device=device)
        #     alpha = torch.scatter(torch.zeros(loss_shape, device=device), 0, loss_hyperedge_masks.nonzero().flatten(), alpha)

    # l = ( alpha * loss_hyperedge ) + ( (1-alpha) * loss_edge )
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
    