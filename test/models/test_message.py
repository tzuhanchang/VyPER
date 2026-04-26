import pytest
import torch
from pathlib import Path


FIXTURES = Path(__file__).parent.parent / "fixtures"

@pytest.fixture
def example_batch():
    return torch.load(FIXTURES / "batched_graphs.pt", weights_only=False)

@pytest.fixture
def expected_EdgeModel_output():
    return torch.load(FIXTURES / "EdgeModel-exp.pt", weights_only=False)

@pytest.fixture
def expected_NodeModel_output():
    return torch.load(FIXTURES / "NodeModel-exp.pt", weights_only=False)

@pytest.fixture
def expected_GlobalModel_output():
    return torch.load(FIXTURES / "GlobalModel-exp.pt", weights_only=False)

@pytest.fixture
def expected_NeutrinoModel_output():
    return torch.load(FIXTURES / "NeutrinoModel-exp.pt", weights_only=False)


@torch.no_grad
def test_EdgeModel(example_batch, expected_EdgeModel_output):
    from VyPER.models.message import EdgeModel

    model = EdgeModel(d_node=7, d_edge=3, d_glob=8, d_out=2,
                      d_embed=8, dropout=0.)

    # Set internal weights to 1.
    for name, param in model.named_parameters():
        param.fill_(1.)

    out = model(example_batch.x, example_batch.edge_index,
                example_batch.edge_attr, example_batch.u, example_batch.batch)
    assert torch.allclose(out, expected_EdgeModel_output)


@torch.no_grad
def test_NodeModel(example_batch, expected_NodeModel_output):
    from VyPER.models.message import NodeModel

    model = NodeModel(d_node=7, d_edge=3, d_glob=8, d_out=2,
                      d_embed=8, dropout=0.)

    # Set internal weights to 1.
    for name, param in model.named_parameters():
        param.fill_(1.)

    out = model(example_batch.x, example_batch.edge_index,
                example_batch.edge_attr, example_batch.u, example_batch.batch)
    assert torch.allclose(out, expected_NodeModel_output)


@torch.no_grad
def test_GlobalModel(example_batch, expected_GlobalModel_output):
    from VyPER.models.message import GlobalModel

    model = GlobalModel(d_node=7, d_edge=3, d_glob=8, d_out=2,
                        d_embed=8, dropout=0.)

    # Set internal weights to 1.
    for name, param in model.named_parameters():
        param.fill_(1.)

    out = model(example_batch.x, example_batch.edge_index,
                example_batch.edge_attr, example_batch.u, example_batch.batch)
    assert torch.allclose(out, expected_GlobalModel_output)


@torch.no_grad
def test_NeutrinoModel(example_batch, expected_NeutrinoModel_output):
    from VyPER.models.message import NeutrinoModel

    model = NeutrinoModel(d_node=7, d_edge=3, d_glob=8, d_out=2,
                          d_embed=8, dropout=0.)

    # Set internal weights to 1.
    for name, param in model.named_parameters():
        param.fill_(1.)

    out = model(example_batch.x, example_batch.edge_index,
                example_batch.edge_attr, example_batch.u, example_batch.batch,
                example_batch.x_fw_mask, example_batch.edge_fw_mask)
    assert torch.allclose(out[0], expected_NeutrinoModel_output[0])
    assert torch.allclose(out[1], expected_NeutrinoModel_output[1])