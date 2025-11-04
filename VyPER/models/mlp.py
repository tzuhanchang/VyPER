from torch import nn, Tensor


class Mlp(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_hidden: int, depth: int=1,
                 activation: callable=nn.SiLU(), dropout: float=0., bias: bool=True):
        super().__init__()

        self.layers = nn.ModuleList([
            # Input Layer
            nn.Sequential(
                nn.Linear(d_in, d_hidden, bias=bias),
                activation,
                nn.Dropout(p=dropout)
            )
        ] + [
            # Hidden Layer(s)
            nn.Sequential(
                nn.Linear(d_hidden, d_hidden, bias=bias),
                activation,
                nn.Dropout(p=dropout)
            ) for _ in range(depth)
        ] + [
            # Output Layer
            nn.Linear(d_hidden, d_out, bias=bias)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            for module in layer.children():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x