import pytest
import torch
from pathlib import Path


FIXTURES = Path(__file__).parent.parent / "fixtures"
D_MODEL = 8
D_HYPEREDGE = 6

# The untransformed features in `batched_graphs.pt` give embeddings up to O(10^3), so the float32
# outputs differ between CPUs and builds by up to ~3e-5 relative (measured against float64).
RTOL, ATOL = 1e-4, 1e-4

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
    neutrinos, as in :class:`VyPER.models.VyPER`. Latent hyperedges (of size
    :obj:`D_HYPEREDGE`) sum a projection of the features of their nodes."""
    def project(t, seed, d=D_MODEL):
        w = torch.sin(1.3 * torch.arange(1, t.size(1) * d + 1) + seed).reshape(t.size(1), d)
        return t @ w

    x = project(batch.x, 0)
    nu_mask = batch.x_fw_mask.to(torch.bool)
    hyperedge_attr = project(batch.x, 3, D_HYPEREDGE)[batch.hyperedge_index].sum(dim=0)
    return dict(x=x, edge_index=batch.edge_index, edge_attr=project(batch.edge_attr, 1),
                u=project(batch.u, 2), batch=batch.batch,
                nu_ctx=x[nu_mask], nu_batch=batch.batch[nu_mask],
                hyperedge_attr=hyperedge_attr, hyperedge_batch=batch.hyperedge_index_batch)


def _class_attention_block(fill=True):
    from VyPER.models.classification import ClassAttentionBlock
    torch.manual_seed(0)
    model = ClassAttentionBlock(d_embed=D_MODEL, d_feedforward=16, nheads=2, dropout=0.)
    return (_fill(model) if fill else model).eval()


EVENT_CLASSIFICATION_SETTINGS = {
    'with_neutrino':    dict(use_neutrino=True),
    'without_neutrino': dict(use_neutrino=False),
    'with_hyperedge':   dict(use_neutrino=True, use_hyperedge=True),
}

def _event_classification(use_neutrino, use_hyperedge=False):
    from VyPER.models.classification import EventClassification
    return _fill(EventClassification(d_model=D_MODEL, d_out=3, d_feedforward=16, use_neutrino=use_neutrino,
                                     use_hyperedge=use_hyperedge, d_hyperedge=D_HYPEREDGE if use_hyperedge else None,
                                     nheads=2, attn_depth=2, dropout=0.)).eval()


@torch.no_grad
def test_ClassAttentionBlock(example_batch, expected_ClassAttentionBlock_output):
    model = _class_attention_block()
    emb = _embeddings(example_batch)

    out = model(emb['u'], emb['x'], emb['batch'])
    assert out.shape == (example_batch.num_graphs, D_MODEL)
    torch.testing.assert_close(out, expected_ClassAttentionBlock_output, rtol=RTOL, atol=ATOL)


def _reference_class_attention(model, q, kv, index, condition=True):
    r"""Reference :class:`ClassAttentionBlock`: the (conditioned) object tokens are
    padded per event and attended over with `nn.MultiheadAttention` and a key padding mask."""
    from torch_geometric.utils import to_dense_batch

    kv = model.norm_kv(kv)
    if condition:
        shift, scale = model.token_modulation(q).chunk(2, dim=-1)
        kv = kv * (1 + scale[index]) + shift[index]
    dense, is_real = to_dense_batch(kv, index)
    attn = model.attn(model.norm_q(q).unsqueeze(1), dense, dense,
                      key_padding_mask=~is_real, need_weights=False)[0].squeeze(1)
    ref = q + attn
    return ref + model.mlp(model.norm_mlp(ref))


@torch.no_grad
def test_ClassAttentionBlock_matches_MultiheadAttention(example_batch):
    model = _class_attention_block()
    emb = _embeddings(example_batch)
    q, kv, index = emb['u'], emb['x'], emb['batch']

    torch.testing.assert_close(model(q, kv, index), _reference_class_attention(model, q, kv, index),
                               rtol=RTOL, atol=ATOL)


@torch.no_grad
def test_ClassAttentionBlock_conditioning_starts_at_zero(example_batch):
    # The modulation is initialised to zero, so a freshly initialised block
    # attends over the unmodulated object tokens
    model = _class_attention_block(fill=False)
    emb = _embeddings(example_batch)
    q, kv, index = emb['u'], emb['x'], emb['batch']

    torch.testing.assert_close(model(q, kv, index), _reference_class_attention(model, q, kv, index, condition=False),
                               rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('setting', EVENT_CLASSIFICATION_SETTINGS)
@torch.no_grad
def test_EventClassification(example_batch, expected_EventClassification_output, setting):
    model = _event_classification(**EVENT_CLASSIFICATION_SETTINGS[setting])
    emb = _embeddings(example_batch)

    out = model(**emb)
    assert out.shape == (example_batch.num_graphs, 3)
    torch.testing.assert_close(out, expected_EventClassification_output[setting], rtol=RTOL, atol=ATOL)


@torch.no_grad
def test_EventClassification_events_are_independent(example_batch):
    from torch_geometric.data import Batch

    model = _event_classification(use_neutrino=True, use_hyperedge=True)
    events = example_batch.to_data_list()

    # Each event in a batch is classified as if it were alone, whatever the batch order
    def batch(data_list):
        return Batch.from_data_list(data_list, follow_batch=['hyperedge_index'])
    batched = model(**_embeddings(batch([events[1], events[0], events[1]])))
    alone = torch.cat([model(**_embeddings(batch([e]))) for e in (events[1], events[0], events[1])])
    torch.testing.assert_close(batched, alone, rtol=RTOL, atol=ATOL)
    assert not torch.allclose(batched[0], batched[1])
