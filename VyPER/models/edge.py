from torch import nn, Tensor
from .mlp import Mlp


class EdgeBlock(nn.Module):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            d_hidden: int,
            depth: int = 2,
            dropout: float = 0.01,
            activation: callable = nn.ReLU(),
            bias: bool = True
        ) -> None:
        super().__init__()

        self.mlp = Mlp(
            d_in=d_in,
            d_out=d_out,
            d_hidden=d_hidden,
            depth=depth,
            activation=activation,
            dropout=dropout,
            bias=bias
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.mlp.reset_parameters()

    def forward(self, edge_attr) -> Tensor:
        return self.mlp(edge_attr)