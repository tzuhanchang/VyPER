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

@torch.no_grad
def test_DiTBlock_key_padding_mask():
    from VyPER.models.attention import DiTBlock

    torch.manual_seed(0)
    model = DiTBlock(d_embed=4, num_heads=1, mlp_ratio=1., dropout=0.).eval()

    # [L, B, d_embed]; the last position of the second sequence is padding
    x = torch.rand(3, 2, 4)
    c = torch.rand(3, 2, 4)
    key_padding_mask = torch.tensor([[False, False, False],
                                     [False, False, True]])

    out = model(x, c, key_padding_mask)

    # Changing the padded position must not change the non-padded outputs
    x_alt = x.clone()
    x_alt[2, 1] = 1e4
    out_alt = model(x_alt, c, key_padding_mask)
    assert torch.allclose(out[:, 0], out_alt[:, 0])
    assert torch.allclose(out[:2, 1], out_alt[:2, 1])


def _run_denoiser(model, nu_batch, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(nu_batch)
    x = torch.rand(n, model.d_x, generator=g)
    c = torch.rand(n, 5, generator=g)
    T = torch.rand(n, generator=g)
    return model(x, c, T, torch.tensor(nu_batch, dtype=torch.long)), (x, c, T)


@torch.no_grad
def test_Denoiser_variable_neutrinos():
    from VyPER.models.attention import Denoiser

    torch.manual_seed(0)
    model = Denoiser(d_embed=8, depth=2, num_heads=1, d_x=3, d_ctx=5).eval()

    # Events with 2 and 3 neutrinos in the same batch
    out, (x, c, T) = _run_denoiser(model, [0, 0, 1, 1, 1])
    assert out.shape == (5, 3)
    assert not out.isnan().any()

    # Each event's output does not depend on the other events in the batch
    out_0 = model(x[:2], c[:2], T[:2], torch.tensor([0, 0]))
    out_1 = model(x[2:], c[2:], T[2:], torch.tensor([1, 1, 1]))
    assert torch.allclose(out, torch.cat([out_0, out_1]), atol=1e-6)


@torch.no_grad
def test_Denoiser_events_without_neutrinos():
    from VyPER.models.attention import Denoiser

    torch.manual_seed(0)
    model = Denoiser(d_embed=8, depth=2, num_heads=1, d_x=3, d_ctx=5).eval()

    # Event 1 (middle) and event 3 (last) have no neutrinos
    out, (x, c, T) = _run_denoiser(model, [0, 0, 2, 2, 2])
    ref, _ = _run_denoiser(model, [0, 0, 1, 1, 1])
    assert out.shape == (5, 3)
    assert torch.allclose(out, ref)

    # No neutrinos in the whole batch
    out = model(torch.rand(0, 3), torch.rand(0, 5), torch.rand(0), torch.zeros(0, dtype=torch.long))
    assert out.shape == (0, 3)
