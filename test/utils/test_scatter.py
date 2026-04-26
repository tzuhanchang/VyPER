import torch


def test_group_batch():
    from VyPER.utils import group_batch

    src = torch.randn(6, 4)
    index = torch.tensor([0, 0, 0, 1, 2, 2])

    expected = torch.full((3, 3, 4), float("-inf"))
    expected[0, :3, :] = src[:3]
    expected[1, :1, :] = src[3]
    expected[2, :2, :] = src[4:]

    out, mask = group_batch(src, index, dim=0, pad_size=3,
                            pad_value=float("-inf"), return_mask=True)
    assert torch.equal(out, expected)
    assert torch.equal(out[mask].reshape(src.size()), src)


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