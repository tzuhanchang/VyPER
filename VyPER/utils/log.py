import torch

import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from torch_hep.lorentz import MomentumTensor
from matplotlib.figure import Figure
from typing import List, Optional


def get_neutrino_p4(
    nu: Tensor,
    reverse_transform_methods: Optional[List[callable]]
) -> MomentumTensor:
    r"""Get neutrino four momentum. If :obj:`reverse_transform_methods` is
    provided, neutrino is unscaled with the list of methods.

    Args:
        nu (Tensor): Neutrino output (Px,Py,Pz).
        reverse_transform_methods (optional, List[callable]): List of reverse
            transformation methods (lambda functions).

    :rtype: :class:`MomentumTensor`
    """
    assert nu.size(1) == 3
    if reverse_transform_methods is not None:
        for column in range(nu.size(1)):
            nu[:,column] = reverse_transform_methods[column](nu[:,column])

    e = torch.sqrt((nu[:,0]*nu[:,0]) + (nu[:,1]*nu[:,1]) + (nu[:,2]*nu[:,2]))
    p = MomentumTensor(torch.cat([e.view(-1,1),
                                  nu[:,0].view(-1,1),
                                  nu[:,1].view(-1,1),
                                  nu[:,2].view(-1,1)], dim=1))
    return p


def log_hist2D(x: np.array, y: np.array, bins: int, range: List) -> Figure:
    r"""Log 2D histogram with Tensorboard.

    Args:
        x (numpy.array): An array containing the x coordinates of 
            the points to be histogrammed.
        y (numpy.array): An array containing the y coordinates of 
            the points to be histogrammed.
        bins (int): Number of bins for the two dimensions.
        range (List): The leftmost and rightmost edges of the bins
            along each dimension.

    :rtype: :class:`matplotlib.figure.Figure`
    """
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.hist2d(x, y, bins=bins, range=range)
    ax.set_axis_off()
    fig.patch.set_visible(False)
    return fig