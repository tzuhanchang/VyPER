import pytest
import torch
from pathlib import Path


FIXTURES = Path(__file__).parent.parent / "fixtures"
D_MODEL = 8

@pytest.fixture
def example_batch():
    return torch.load(FIXTURES / "batched_graphs.pt", weights_only=False)

@pytest.fixture
def expected_ClassAttentionBlock_output():
    return torch.load(FIXTURES / "ClassAttentionBlock-exp.pt", weights_only=False)

@pytest.fixture
def expected_EventClassification_output():
    return torch.load(FIXTURES / "EventClassification-exp.pt", weights_only=False)


def _fill(model):
    r"""Set internal weights to a fixed sine pattern. Unlike constant weights,
    these give distinct object tokens after the token projections and layer
    norms, so the attention over them is not degenerate."""
    with torch.no_grad():
        for i, (name, param) in enumerate(model.named_parameters()):
            param.copy_(torch.sin(0.7 * torch.arange(1, param.numel() + 1) + i).reshape(param.shape))
    return model


def _embeddings(batch):
    r"""Fixed projections of the node, edge and global features of
    :obj:`batch` to :obj:`D_MODEL`, standing in for the message-passing
    embeddings. Neutrino contexts are taken from the nodes associated with
    neutrinos, as in :class:`VyPER.models.VyPER`."""
    def project(t, seed):
        w = torch.sin(1.3 * torch.arange(1, t.size(1) * D_MODEL + 1) + seed).reshape(t.size(1), D_MODEL)
        return t @ w

    x = project(batch.x, 0)
    nu_mask = batch.x_fw_mask.to(torch.bool)
    return dict(x=x, edge_index=batch.edge_index, edge_attr=project(batch.edge_attr, 1),
                u=project(batch.u, 2), batch=batch.batch,
                nu_ctx=x[nu_mask], nu_batch=batch.batch[nu_mask])


def _class_attention_block():
    from VyPER.models.classification import ClassAttentionBlock
    return _fill(ClassAttentionBlock(d_embed=D_MODEL, d_feedforward=16, nheads=2, dropout=0.)).eval()


EVENT_CLASSIFICATION_SETTINGS = {
    'with_neutrino':    dict(use_neutrino=True,  use_edge=True),
    'without_neutrino': dict(use_neutrino=False, use_edge=True),
    'without_edge':     dict(use_neutrino=True,  use_edge=False),
}

def _event_classification(use_neutrino, use_edge):
    from VyPER.models.classification import EventClassification
    return _fill(EventClassification(d_model=D_MODEL, d_out=3, d_feedforward=16, use_neutrino=use_neutrino,
                                     use_edge=use_edge, nheads=2, attn_depth=2, dropout=0.)).eval()


@torch.no_grad
def test_ClassAttentionBlock(example_batch, expected_ClassAttentionBlock_output):
    model = _class_attention_block()
    emb = _embeddings(example_batch)

    out = model(emb['u'], emb['x'], emb['batch'])
    assert out.shape == (example_batch.num_graphs, D_MODEL)
    assert torch.allclose(out, expected_ClassAttentionBlock_output, atol=1e-6)


@torch.no_grad
def test_ClassAttentionBlock_matches_MultiheadAttention(example_batch):
    from torch_geometric.utils import to_dense_batch

    model = _class_attention_block()
    emb = _embeddings(example_batch)
    q, kv, index = emb['u'], emb['x'], emb['batch']

    # Reference: tokens padded per event and `nn.MultiheadAttention` with a key padding mask
    dense, is_real = to_dense_batch(model.norm_kv(kv), index)
    attn = model.attn(model.norm_q(q).unsqueeze(1), dense, dense,
                      key_padding_mask=~is_real, need_weights=False)[0].squeeze(1)
    ref = q + attn
    ref = ref + model.mlp(model.norm_mlp(ref))

    assert torch.allclose(model(q, kv, index), ref, atol=1e-6)


@pytest.mark.parametrize('setting', EVENT_CLASSIFICATION_SETTINGS)
@torch.no_grad
def test_EventClassification(example_batch, expected_EventClassification_output, setting):
    model = _event_classification(**EVENT_CLASSIFICATION_SETTINGS[setting])
    emb = _embeddings(example_batch)

    out = model(**emb)
    assert out.shape == (example_batch.num_graphs, 3)
    assert torch.allclose(out, expected_EventClassification_output[setting], atol=1e-6)


@torch.no_grad
def test_EventClassification_events_are_independent(example_batch):
    from torch_geometric.data import Batch

    model = _event_classification(use_neutrino=True, use_edge=True)
    events = example_batch.to_data_list()

    # Each event in a batch is classified as if it were alone, whatever the batch order
    batched = model(**_embeddings(Batch.from_data_list([events[1], events[0], events[1]])))
    alone = torch.cat([model(**_embeddings(Batch.from_data_list([e]))) for e in (events[1], events[0], events[1])])
    assert torch.allclose(batched, alone, atol=1e-6)
    assert not torch.allclose(batched[0], batched[1])
