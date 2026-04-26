import pytest
import torch
from pathlib import Path

from VyPER.data import VyPERDataset


FIXTURES = Path(__file__).parent.parent / "fixtures"

@pytest.fixture
def example_graph():
    return torch.load(FIXTURES / "graph.pt", weights_only=False)

@pytest.fixture
def example_dataset():
    root = FIXTURES / "test.h5"
    config = FIXTURES / "config.yaml"
    return VyPERDataset(root=root, config=config, training=True)


def test_VyPERDataset(example_graph, example_dataset):
    data = example_dataset[0]
    for key, tensor in example_graph.items():
        assert torch.allclose(tensor, data[key])
