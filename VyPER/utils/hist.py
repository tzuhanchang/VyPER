import numpy as np
import matplotlib.pyplot as plt

from typing import List

def log_histogram(x: np.array, y: np.array, bins: int, range: List):
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.hist2d(x, y, bins=bins, range=range)
    ax.set_axis_off()
    fig.patch.set_visible(False)
    return fig