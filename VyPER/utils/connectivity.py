import torch

from torch import Tensor
from torch_geometric.utils import degree
from itertools import combinations
from typing import Union, Tuple, List, Optional


def edge_reduction(src: Tensor, index: Tensor, reduction='mean', num_nodes: Optional[int]=None,
                   return_reduced_indices: bool=True) -> Union[Tensor,Tuple[Tensor,Tensor]]:
    r"""Reduce a directed-graph into a undirected-graph by merging two
    directed-edges who share the same endpoints into an undirected edge.

    Args:
        src (Tensor): Edge tensor.
        index (Tensor): Edge index tensor.
        reduction (optional, str): Reduction methods.
            (default :obj:`str`=None)
        num_nodes (optional, int): Number of nodes in the graph. (default :obj:`None`)
        return_reduced_indices (optional, bool): Returning reduced `edge_index`.
            (default :obj:`bool`=True)

    :rtype: :class:`Union[Tensor,Tuple[Tensor,Tensor]]`
    """
    if num_nodes is None:
        num_nodes = int(torch.max(index)) + 1

    edge_index_comb = torch.tensor(list(combinations(range(num_nodes),r=2)),
                                   device=src.device).transpose(0,1)
    indices = torch.all(edge_index_comb.unsqueeze(2) == index.unsqueeze(1),
                        dim=0).nonzero()[:,1]
    indices_flip = torch.all(edge_index_comb.flip(0).unsqueeze(2) == index.unsqueeze(1),
                        dim=0).nonzero()[:,1]
    src_i = src[indices,:]
    src_j = src[indices_flip,:]

    out = torch.cat([src_i.unsqueeze(2),src_j.unsqueeze(2)], dim=2)

    if reduction.lower() == 'mean':
        out = out.mean(dim=2)
    elif reduction.lower() == 'sum':
        out = out.min(dim=2)
    elif reduction.lower() == 'max':
        out = out.max(dim=2)[0]
    elif reduction.lower() == 'min':
        out = out.min(dim=2)[0]
    else:
        raise NotImplementedError(f"{reduction} reduction meethod is not available.")

    return (out, edge_index_comb) if return_reduced_indices else out


def unbatch_hyperedge_index(hyperedge_index: Tensor, batch: Tensor) -> List[Tensor]:
    r"""Splits the :obj:`hyperedge_index` according to a :obj:`batch` vector.

    Args:
        hyperedge_index (Tensor): Hyperedge index tensor.
        batch (Tensor): The batch vector.

    :rtype: :class:`List[Tensor]`
    """
    out = torch.split_with_sizes(hyperedge_index, degree(batch).to(torch.int64).tolist(), dim=1)
    return [x-torch.min(x) for x in out]