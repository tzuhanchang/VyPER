import math
import pytest
import torch

from VyPER.models.diffusion import NeutrinoDiffusion, TimeGrid


def _diffusion(**kwargs):
    torch.manual_seed(0)
    return NeutrinoDiffusion(d_ctx=4, d_embed=8, d_target=3, num_heads=2, num_message_steps=2,
                             num_dit_blocks=2, num_sampling_steps=4, **kwargs).eval()


def _inputs(nu_batch, d_ctx=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    nu_batch = torch.tensor(nu_batch, dtype=torch.long)
    return torch.rand(len(nu_batch), d_ctx, generator=g), nu_batch


def _uniform(num_steps):
    return 1. - torch.arange(num_steps + 1, dtype=torch.float32) / num_steps


@pytest.mark.parametrize("cfg", [dict(), dict(scheme='power', rho=3.), dict(scheme='power', rho=0.5),
                                 dict(scheme='cosine'), dict(scheme='beta', a=0.4, b=2.5)])
@pytest.mark.parametrize("num_steps", [1, 2, 7, 50])
def test_grid_is_decreasing_from_one_to_zero(cfg, num_steps):
    T = TimeGrid(**cfg).grid(num_steps)
    assert T.shape == (num_steps + 1,) and T.dtype == torch.float32
    assert T[0] == 1. and T[-1] == 0.
    assert (T[1:] < T[:-1]).all()


def test_grid_schemes():
    u = _uniform(10)
    torch.testing.assert_close(TimeGrid().grid(10), u)
    torch.testing.assert_close(TimeGrid(scheme='power', rho=2.).grid(10), u**2)
    # Beta(1, 1) is uniform, and Beta(1/2, 1/2) (arcsine) quantiles are the cosine grid
    torch.testing.assert_close(TimeGrid(scheme='beta').grid(10), u, atol=1e-5, rtol=0)
    torch.testing.assert_close(TimeGrid(scheme='beta', a=.5, b=.5).grid(10), TimeGrid(scheme='cosine').grid(10),
                               atol=1e-4, rtol=0)
    # Beta(2, 1): CDF T^2
    torch.testing.assert_close(TimeGrid(scheme='beta', a=2., b=1.).grid(10), u.sqrt(), atol=1e-4, rtol=0)


def test_grid_single_step_matches_grid():
    grid = TimeGrid(scheme='beta', a=.7, b=1.3)
    N = torch.tensor(9)
    single = torch.stack([grid(torch.tensor(i), N) for i in range(10)])
    torch.testing.assert_close(single, grid.grid(9))


def test_grid_invalid():
    with pytest.raises(NotImplementedError):
        TimeGrid(scheme='linear')
    with pytest.raises(ValueError):
        TimeGrid(scheme='beta', a=0.)
    with pytest.raises(RuntimeError):
        TimeGrid(scheme='adaptive').grid(4)


def test_grid_is_not_saved():
    model = _diffusion(sampling_schedule=dict(scheme='beta', a=.5, b=.5))
    assert not any('time_grid' in key for key in model.state_dict())
    # Checkpoints load into models with any schedule
    _diffusion().load_state_dict(model.state_dict())


def test_set_density():
    grid = TimeGrid(scheme='adaptive', uniform_mix=0.)
    knots = torch.linspace(0, 1, 5, dtype=torch.float64)
    # All steps in the lower half
    grid.set_density(knots, torch.tensor([1., 1., 0., 0.]))
    torch.testing.assert_close(grid.grid(4), torch.tensor([1., .375, .25, .125, 0.]))


@torch.no_grad()
def test_uniform_solver_matches_constant_step():
    model = _diffusion()
    ctx, nu_batch = _inputs([0, 0, 1, 3, 3, 3])
    noise = model.sample_noise(len(nu_batch))

    # Constant dT
    state = model.Denoiser.prepare_sampling(ctx, nu_batch)
    T, dT, D = torch.ones(()), 1 / 4, model.Denoiser.to_dense(noise, state)
    for _ in range(4):
        D = model._step(T, T - dT, D, state)
        T = T - dT

    torch.testing.assert_close(model.solver(ctx, nu_batch, noise=noise), model.Denoiser.from_dense(D, state))


@torch.no_grad()
@pytest.mark.parametrize("cfg", [dict(scheme='power', rho=2.), dict(scheme='cosine'), dict(scheme='beta', a=.7, b=.7),
                                 dict(scheme='adaptive', calibration_steps=20)])
def test_solver_schemes(cfg):
    model = _diffusion(sampling_schedule=cfg)
    ctx, nu_batch = _inputs([0, 0, 1, 3, 3, 3])
    noise = model.sample_noise(len(nu_batch))
    out = model.solver(ctx, nu_batch, noise=noise, num_steps=5)
    assert out.shape == (6, 3) and torch.isfinite(out).all()
    assert not model.time_grid.needs_calibration
    # Same steps with a tensor number of steps
    torch.testing.assert_close(model.solver(ctx, nu_batch, noise=noise, num_steps=torch.tensor(5)), out)
    # Events stay independent
    out_0 = model.solver(ctx[:3], nu_batch[:3], noise=noise[:3], num_steps=5)
    torch.testing.assert_close(out[:3], out_0, atol=1e-5, rtol=1e-5)
    # Differs from the uniform grid
    assert not torch.allclose(out, _diffusion().solver(ctx, nu_batch, noise=noise, num_steps=5))


@torch.no_grad()
def test_adaptive_calibration():
    model = _diffusion(sampling_schedule=dict(scheme='adaptive', calibration_steps=40, calibration_size=4))
    ctx, nu_batch = _inputs([0, 0, 1, 3, 3, 3])
    # No neutrinos: nothing to calibrate on
    assert model.solver(ctx[:0], nu_batch[:0]).shape == (0, 3)
    assert model.time_grid.needs_calibration

    kappa = model.calibrate_time_grid(ctx, nu_batch)
    assert kappa.shape == (40,) and (kappa >= 0).all()
    T_mid, density = model.time_grid.density
    # Normalised density, at least `uniform_mix` everywhere
    assert math.isclose((density / 40).sum().item(), 1., rel_tol=1e-9)
    assert (density >= model.time_grid.uniform_mix - 1e-12).all()
    T = model.time_grid.grid(10)
    assert T[0] == 1. and T[-1] == 0. and (T[1:] < T[:-1]).all()


@torch.no_grad()
def test_onnx_export_adaptive(tmp_path):
    pytest.importorskip("onnxscript")
    onnxruntime = pytest.importorskip("onnxruntime")
    from VyPER.models import VyPER
    from VyPER.onnx import VyPERONNX, export_onnx, calibrate_time_grid

    torch.manual_seed(0)
    model = VyPER(node_in_channels=7, edge_in_channels=3, global_in_channels=8, edge_out_channels=3,
                  hyperedge_out_channels=2, nu_out_channels=3, message_feats=8, attn_feats=8,
                  num_attn_heads=2, num_message_layers=2, num_dit_blocks=1, hyperedge_feats=8,
                  num_sampling_steps=3, sampling_schedule=dict(scheme='adaptive', calibration_steps=50)).eval()
    wrapper = VyPERONNX(model).eval()
    data = torch.load('test/fixtures/batched_graphs.pt', weights_only=False)

    inputs = wrapper.example_inputs(data, num_sampling_steps=3, noise_as_input=True)
    with pytest.raises(RuntimeError, match="Calibrate"):
        export_onnx(wrapper, inputs, str(tmp_path / "VyPER.onnx"), opset_version=21)

    calibrate_time_grid(wrapper, data)
    export_onnx(wrapper, inputs, str(tmp_path / "VyPER.onnx"), opset_version=21)
    session = onnxruntime.InferenceSession(str(tmp_path / "VyPER.onnx"), providers=["CPUExecutionProvider"])
    for steps in (3, 7):
        inputs = wrapper.example_inputs(data, num_sampling_steps=steps, noise_as_input=True)
        ort_out = session.run(None, {k: v.numpy() for k, v in inputs.items()})
        for pt, ort in zip(wrapper(**inputs), ort_out):
            torch.testing.assert_close(torch.from_numpy(ort), pt, rtol=1e-4, atol=1e-4)
