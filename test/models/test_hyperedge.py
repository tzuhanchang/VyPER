import pytest
import torch
from pathlib import Path


FIXTURES = Path(__file__).parent.parent / "fixtures"

@pytest.fixture
def example_batch():
    return torch.load(FIXTURES / "batched_graphs.pt", weights_only=False)

@pytest.fixture
def expected_HyperedgeBlock_output():
    return torch.load(FIXTURES / "HyperedgeBlock-exp.pt", weights_only=False)


@torch.no_grad
def test_HyperedgeBlock(example_batch, expected_HyperedgeBlock_output):
    from VyPER.models.hyperedge import HyperedgeBlock

    model = HyperedgeBlock(node_in_channels=7, node_out_channels=2,
                           global_in_channels=8, message_feats=8, dropout=0.)

    # Set internal weights to 1.
    for name, param in model.named_parameters():
        param.fill_(1.)

    out = model(example_batch.x, example_batch.u, example_batch.batch,
                example_batch.hyperedge_index, example_batch.hyperedge_index_batch, r=3)
    assert torch.allclose(out[0], expected_HyperedgeBlock_output[0])
    assert torch.allclose(out[1], expected_HyperedgeBlock_output[1])