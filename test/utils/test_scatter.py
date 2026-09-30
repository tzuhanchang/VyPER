import torch


def test_softmax():
    from VyPER.utils import softmax

    src = torch.tensor([1., 1., 1., 1.])
    index = torch.tensor([0, 0, 1, 2])

    out = softmax(src, index, dim_size=4)
    assert out.tolist() == [0.5, 0.5, 1, 1]

    src = src.view(-1, 1)
    out = softmax(src, index, dim_size=4)
    assert out.tolist() == [[0.5], [0.5], [1], [1]]

    jit = torch.jit.script(softmax)
    assert torch.allclose(jit(src, index, dim_size=4), out)