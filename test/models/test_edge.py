import pytest
import torch
from pathlib import Path


FIXTURES = Path(__file__).parent.parent / "fixtures"

@pytest.fixture
def example_batch():
    return torch.load(FIXTURES / "batched_graphs.pt", weights_only=False)

@pytest.fixture
def expected_EdgeBlock_output():
    return torch.load(FIXTURES / "EdgeBlock-exp.pt", weights_only=False)


def _edge_block():
    from VyPER.models.edge import EdgeBlock

    model = EdgeBlock(d_in=3, d_out=3, d_hidden=8, depth=2, dropout=0.).eval()

    # Set internal weights to 1.
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.fill_(1.)
    return model


@torch.no_grad
def test_EdgeBlock(example_batch, expected_EdgeBlock_output):
    model = _edge_block()

    out = model(example_batch.edge_attr)
    assert out.shape == (example_batch.edge_attr.size(0), 3)
    assert torch.allclose(out, expected_EdgeBlock_output)
