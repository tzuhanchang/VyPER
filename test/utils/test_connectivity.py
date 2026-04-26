import pytest
import torch
from itertools import combinations


@pytest.mark.parametrize('reduction', ['mean','sum','max','min'])
def test_edge_reduction(reduction):
    from VyPER.utils import edge_reduction

    src = torch.tensor([[0.5,0.5],[0.1,0.9],
                        [0.3,0.7],[0.4,0.6],
                        [0.2,0.8],[0.9,0.1],
                        [0.9,0.1],[0.8,0.2]])
    index = torch.tensor([[0,0,1,1,2,2],[1,2,0,2,0,1]])

    out, reduced_index = edge_reduction(src, index,
                            reduction=reduction,
                            return_reduced_indices=True)

    assert reduced_index.tolist() == [[0,0,1],[1,2,2]]

    if reduction == 'mean':
        t1 = torch.vstack([src[0],src[2]]).mean(0)
        t2 = torch.vstack([src[1],src[4]]).mean(0)
        t3 = torch.vstack([src[3],src[5]]).mean(0)
        assert torch.allclose(out, torch.vstack([t1,t2,t3]))
    if reduction == 'sum':
        t1 = torch.vstack([src[0],src[2]]).sum(0)
        t2 = torch.vstack([src[1],src[4]]).sum(0)
        t3 = torch.vstack([src[3],src[5]]).sum(0)
        assert torch.allclose(out, torch.vstack([t1,t2,t3]))
    if reduction == 'max':
        t1 = torch.vstack([src[0],src[2]]).max(0)[0]
        t2 = torch.vstack([src[1],src[4]]).max(0)[0]
        t3 = torch.vstack([src[3],src[5]]).max(0)[0]
        assert torch.allclose(out, torch.vstack([t1,t2,t3]))
    if reduction == 'min':
        t1 = torch.vstack([src[0],src[2]]).min(0)[0]
        t2 = torch.vstack([src[1],src[4]]).min(0)[0]
        t3 = torch.vstack([src[3],src[5]]).min(0)[0]
        assert torch.allclose(out, torch.vstack([t1,t2,t3]))


def test_unbatch_hyperedge_index():
    from VyPER.utils import unbatch_hyperedge_index

    h1 = torch.tensor(list(combinations(range(0,4), r=3)))
    h2 = torch.tensor(list(combinations(range(4,7), r=3)))

    src = torch.cat([h1,h2], dim=0).transpose(0,1)
    index = torch.cat([torch.zeros(len(h1), dtype=torch.int64),
                       torch.ones(len(h2), dtype=torch.int64)])

    out = unbatch_hyperedge_index(src, index)
    assert torch.allclose(out[0].transpose(0,1), h1)
    assert torch.allclose(out[1].transpose(0,1), h2-4)