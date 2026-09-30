import pytest
import torch


def _diffusion(num_sampling_steps=4):
    from VyPER.models.diffusion import NeutrinoDiffusion

    torch.manual_seed(0)
    return NeutrinoDiffusion(d_ctx=4, d_embed=8, d_target=3, num_heads=2, num_message_steps=2,
                             num_dit_blocks=2, num_sampling_steps=num_sampling_steps).eval()


def _inputs(nu_batch, d_ctx=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    nu_batch = torch.tensor(nu_batch, dtype=torch.long)
    return torch.rand(len(nu_batch), d_ctx, generator=g), nu_batch


@torch.no_grad
def test_Denoiser_sample_step():
    model = _diffusion().Denoiser
    ctx, nu_batch = _inputs([0, 0, 2, 2, 2, 5])
    x = torch.rand(len(nu_batch), 3)
    T = torch.tensor(0.37)

    state = model.prepare_sampling(ctx, nu_batch)
    out = model.from_dense(model.sample_step(model.to_dense(x, state), T, state), state)
    ref = model(x, ctx, T.expand(len(nu_batch)), nu_batch)
    assert torch.allclose(out, ref, atol=1e-6)


@torch.no_grad
def test_solver_steps_as_tensor():
    model = _diffusion()
    ctx, nu_batch = _inputs([0, 0, 1, 3, 3, 3])
    noise = model.sample_noise(len(nu_batch))

    out_int = model.solver(ctx, nu_batch, noise=noise, num_steps=4)
    out_tensor = model.solver(ctx, nu_batch, noise=noise, num_steps=torch.tensor(4))
    assert out_int.shape == (6, 3)
    assert torch.allclose(out_int, out_tensor)

    # Default number of steps
    assert torch.allclose(model.solver(ctx, nu_batch, noise=noise), out_int)


@torch.no_grad
def test_solver_events_are_independent():
    model = _diffusion()
    ctx, nu_batch = _inputs([0, 0, 1, 1, 1])
    noise = model.sample_noise(len(nu_batch))

    out = model.solver(ctx, nu_batch, noise=noise)
    out_0 = model.solver(ctx[:2], nu_batch[:2], noise=noise[:2])
    out_1 = model.solver(ctx[2:], nu_batch[2:], noise=noise[2:])
    assert torch.allclose(out, torch.cat([out_0, out_1]), atol=1e-5)


@torch.no_grad
def test_solver_without_neutrinos():
    model = _diffusion()
    ctx, nu_batch = _inputs([])
    assert model.solver(ctx, nu_batch).shape == (0, 3)
    assert model.solver(ctx, nu_batch, num_steps=torch.tensor(3)).shape == (0, 3)

