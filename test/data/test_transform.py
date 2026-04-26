import pytest
import torch
import math
from pathlib import Path


FIXTURES = Path(__file__).parent.parent / "fixtures"

@pytest.fixture
def example_graph():
    return torch.load(FIXTURES / "graph.pt", weights_only=False)


def test_TransformFeatures(example_graph):
    from VyPER.data.transform import TransformFeatures
    transform = TransformFeatures(['x'],
                                  [[lambda x: torch.log(x),
                                    lambda x: x / math.pi,
                                    lambda x: x / math.pi,
                                    lambda x: torch.log(x),
                                    lambda x: x,
                                    lambda x: x]])

    out = transform.forward(example_graph)

    val = example_graph.x
    val[:,0] = torch.log(val[:,0])
    val[:,1] = val[:,1] / math.pi
    val[:,2] = val[:,2] / math.pi
    val[:,3] = torch.log(val[:,3])

    assert torch.allclose(out.x, val)