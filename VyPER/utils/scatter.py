import torch
from torch import Tensor
from torch_geometric.utils import group_cat, degree
from typing import Union, Optional, Tuple

def group_batch(
        src: Tensor, index: Tensor, dim: int = 0,
        pad_size: Optional[int] = None, pad_value: float = float("-inf"),
        return_mask: bool = False) -> Union[Tuple[Tensor, Tensor], Tensor]:
    r"""Tensor :obj:`src` is batched into groups according to :obj:`index`
    in the given dimension :obj:`dim`.
    Padding is applied to enforce consistent size of batched groups.

    Args:
        src (torch.Tensor): The srouce tensor.
        index (torch.Tensor): The index tensor.
        dim (int, optional): The dimension along which to batch.
            (default: :obj:`0`)
        pad_size (int, optional): The size of the batch. If not specified,
            will use the largest computed (unweighted) degree of :obj:`index`.
            (default: :obj:`None`)
        pad_value (float, optional): The fill value used for padding.
            (default: :obj:`float("-inf")`)
        return_mask (bool, optional): If true, will return a tensor with masks
            indicating which elements within batched tensor are not padding.
            (default: :obj:`False`)

    Example:
        >>> src = tensor([[0.1349, 0.8266],
        ...               [0.3651, 0.1737],
        ...               [0.0211, 0.7000],
        ...               [0.1971, 0.2903],
        ...               [0.4086, 0.7221]])
        >>> index = torch.LongTensor([0,0,1,2,2])
        >>> group_batch(src, index, dim=0, pad_size=3, return_mask=True)
        (tensor([[[0.1349, 0.8266],
                  [0.3651, 0.1737],
                  [  -inf,   -inf]],
                 [[0.0211, 0.7000],
                  [  -inf,   -inf],
                  [  -inf,   -inf]],
                 [[0.1971, 0.2903],
                  [0.4086, 0.7221],
                  [  -inf,   -inf]]])
        tensor([[[ True,  True],
                 [ True,  True],
                 [False, False]],
                [[ True,  True],
                 [False, False],
                 [False, False]],
                [[ True,  True],
                 [ True,  True],
                 [False, False]]]))
    """
    device = src.device
    d = degree(index)
    pad_size = max(d) if pad_size is None else pad_size
    degree_missing_in_dim = (pad_size - d).to(torch.long)
    pad_index = torch.unique(index).repeat_interleave(degree_missing_in_dim)

    def batching(  # type: ignore
            input, fill):
        padding_fill = torch.full(input.shape, fill,
                                  device=device).index_select(dim, pad_index)
        padded = group_cat([input, padding_fill], [index, pad_index], dim)
        return torch.stack(padded.split(pad_size, dim), dim)

    out = batching(src, pad_value)
    return (out, batching(torch.full(src.shape, True, device=device),
                          False)) if return_mask else out