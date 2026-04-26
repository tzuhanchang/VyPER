import torch


@torch.no_grad
def test_DiTBlock():
    from VyPER.models.attention import DiTBlock

    x = torch.tensor([[[0.8951, 0.1673, 0.3210],
                       [0.5729, 0.6024, 0.9826],
                       [0.7987, 0.1295, 0.9035]],
                      [[0.3351, 0.9891, 0.0147],
                       [0.1112, 0.0032, 0.7806],
                       [0.9388, 0.4658, 0.1457]]])
    c = torch.tensor([[[0.8975, 0.5197, 0.9533],
                       [0.8467, 0.2683, 0.8699],
                       [0.5975, 0.1532, 0.9306]],
                      [[0.3234, 0.9646, 0.8054],
                       [0.3497, 0.4051, 0.5562],
                       [0.3741, 0.3835, 0.8836]]])
    y = torch.tensor([[[148.6674, 147.9397, 148.0934],
                       [119.4913, 119.5208, 119.9009],
                       [ 99.8514,  99.1822,  99.9562]],
                      [[131.8627, 132.5166, 131.5423],
                       [ 81.9694,  81.8614,  82.6387],
                       [ 96.0566,  95.5836,  95.2635]]])

    model = DiTBlock(d_embed=x.size(-1), num_heads=1, mlp_ratio=1.,
                     dropout=0.)

    # Set internal weights to 1.
    for name, param in model.named_parameters():
        param.fill_(1.)

    out = model(x, c)
    assert torch.allclose(out, y, rtol=1e-3)


@torch.no_grad
def test_DiTOut():
    from VyPER.models.attention import DiTOut

    x = torch.tensor([[[0.8951, 0.1673, 0.3210],
                       [0.5729, 0.6024, 0.9826],
                       [0.7987, 0.1295, 0.9035]],
                      [[0.3351, 0.9891, 0.0147],
                       [0.1112, 0.0032, 0.7806],
                       [0.9388, 0.4658, 0.1457]]])
    c = torch.tensor([[[0.8975, 0.5197, 0.9533],
                       [0.8467, 0.2683, 0.8699],
                       [0.5975, 0.1532, 0.9306]],
                      [[0.3234, 0.9646, 0.8054],
                       [0.3497, 0.4051, 0.5562],
                       [0.3741, 0.3835, 0.8836]]])
    y = torch.tensor([[[8.9548, 8.9548],
                       [8.0731, 8.0731],
                       [7.4060, 7.4060]],
                      [[8.3278, 8.3278],
                       [6.4050, 6.4050],
                       [7.2248, 7.2248]]])

    model = DiTOut(hidden_size=3, out_channels=2)

    # Set internal weights to 1.
    for name, param in model.named_parameters():
        param.fill_(1.)

    out = model(x, c)
    assert torch.allclose(out, y, rtol=1e-3)