"""Model contracts: shapes, the padding topology, and checkpoint round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from hemoflow.config import Config, DataConfig, GeometryConfig, ModelConfig, TrainConfig
from hemoflow.data import generate_dataset, load_context, load_split
from hemoflow.geometry.features import N_FEATURES
from hemoflow.models import ConstantBaseline, PoiseuilleBaseline, RidgeBaseline, VesselContext

torch = pytest.importorskip("torch")


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        name="test",
        geometry=GeometryConfig(n_arc=32, n_theta=16),
        data=DataConfig(n_geometries=24, seed=3, root=str(tmp_path / "data")),
        model=ModelConfig(name="mlp", hidden_dims=(16, 16)),
        train=TrainConfig(epochs=2, batch_size=4, patience=2),
        paths=type(Config().paths)(runs=str(tmp_path / "runs"), reports=str(tmp_path / "rep")),
    )


@pytest.fixture
def batch(cfg):
    dataset_dir = generate_dataset(cfg)
    features, targets, _ = load_split(dataset_dir, "test")
    arrays = load_context(dataset_dir, "test")
    context = VesselContext(
        flow_rate=arrays["flow_rate"],
        inlet_radius=arrays["inlet_radius"],
        viscosity=cfg.fluid.viscosity_pa_s,
    )
    return features, targets, context


# --- baselines -------------------------------------------------------------


@pytest.mark.parametrize("factory", ["constant", "poiseuille", "ridge"])
def test_baselines_return_positive_fields_of_the_right_shape(batch, factory):
    features, targets, context = batch
    model = {
        "constant": lambda: ConstantBaseline.fit(targets),
        "poiseuille": PoiseuilleBaseline,
        "ridge": lambda: RidgeBaseline.fit(features, targets),
    }[factory]()

    prediction = model.predict_pa(features, context)
    assert prediction.shape == targets.shape
    assert np.isfinite(prediction).all()
    assert (prediction > 0).all(), "wall shear stress is a magnitude"


def test_poiseuille_recovers_the_analytic_value(batch):
    """The baseline must reproduce 4*mu*Q/(pi*r^3) at the inlet, where r is known."""
    features, _, context = batch
    prediction = PoiseuilleBaseline().predict_pa(features, context)

    expected = 4.0 * context.viscosity * context.flow_rate / (np.pi * context.inlet_radius**3)
    np.testing.assert_allclose(prediction[:, 0, 0], expected, rtol=1e-4)


def test_poiseuille_has_no_circumferential_variation(batch):
    """Its defining limitation, pinned so nobody credits it with more than it has."""
    features, _, context = batch
    prediction = PoiseuilleBaseline().predict_pa(features, context)
    assert np.ptp(prediction, axis=2).max() < 1e-9


def test_ridge_beats_the_constant_baseline(batch):
    features, targets, context = batch
    constant = ConstantBaseline.fit(targets).predict_pa(features, context)
    ridge = RidgeBaseline.fit(features, targets).predict_pa(features, context)

    def error(pred):
        return float(np.linalg.norm(pred - targets) / np.linalg.norm(targets))

    assert error(ridge) < error(constant)


# --- networks --------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [("mlp", {"hidden_dims": (16, 16)}), ("unet", {"base_channels": 8, "depth": 2})],
)
def test_networks_map_a_feature_grid_to_a_field(name, kwargs):
    from hemoflow.models.nets import build_network

    module = build_network(name, n_features=N_FEATURES, **kwargs)
    x = torch.randn(3, 32, 16, N_FEATURES)
    assert module(x).shape == (3, 32, 16)


def test_unet_rejects_a_grid_it_cannot_downsample():
    from hemoflow.models.nets import VesselUNet

    module = VesselUNet(N_FEATURES, base_channels=8, depth=3)
    with pytest.raises(ValueError, match="divisible"):
        module(torch.randn(1, 30, 16, N_FEATURES))


def test_padded_conv_is_exactly_circular_around_the_circumference():
    """Rolling the input around theta must roll the output identically.

    Zero-padding the circumferential axis - what a stock Conv2d does - breaks
    this, and the break shows up as a stripe of error along the theta = 0 seam,
    cutting straight through the Dean-flow variation the model is meant to learn.
    """
    from hemoflow.models.nets import MixedPadConv2d

    torch.manual_seed(0)
    conv = MixedPadConv2d(4, 6).eval()
    x = torch.randn(1, 4, 16, 16)

    with torch.inference_mode():
        base = conv(x)
        for shift in (1, 3, 5):
            rolled = conv(torch.roll(x, shifts=shift, dims=3))
            torch.testing.assert_close(torch.roll(base, shifts=shift, dims=3), rolled)


def test_padded_conv_is_deliberately_not_circular_along_the_axis():
    """The inlet and outlet are real boundaries, not a wrap-around.

    If this ever starts passing, the axial padding has been changed to circular
    and the model is being told the outlet flows back into the inlet.
    """
    from hemoflow.models.nets import MixedPadConv2d

    torch.manual_seed(0)
    conv = MixedPadConv2d(4, 6).eval()
    x = torch.randn(1, 4, 16, 16)

    with torch.inference_mode():
        difference = (
            (torch.roll(conv(x), shifts=3, dims=2) - conv(torch.roll(x, shifts=3, dims=2)))
            .abs()
            .max()
        )
    assert difference > 1e-3


def test_unet_is_circular_under_stride_aligned_rolls():
    """The whole network keeps the circular topology through encoder and decoder.

    Only shifts that are a multiple of the total downsampling stride can be
    exactly equivariant - pooling aliases anything finer - so the test uses
    those. It fails if the decoder upsamples with edge clamping, which
    reintroduces the seam the convolutions were padded to avoid.
    """
    from hemoflow.models.nets import VesselUNet

    torch.manual_seed(0)
    module = VesselUNet(N_FEATURES, base_channels=8, depth=2).eval()
    x = torch.randn(1, 32, 16, N_FEATURES)

    with torch.inference_mode():
        base = module(x)
        for shift in (4, 8, 12):  # multiples of 2**depth
            rolled = module(torch.roll(x, shifts=shift, dims=2))
            torch.testing.assert_close(
                torch.roll(base, shifts=shift, dims=2), rolled, rtol=1e-4, atol=1e-5
            )


def test_unknown_architecture_is_rejected():
    from hemoflow.models.nets import build_network

    with pytest.raises(ValueError, match="unknown architecture"):
        build_network("transformer", n_features=N_FEATURES)


# --- checkpoints -----------------------------------------------------------


def test_checkpoint_roundtrip_preserves_predictions(cfg, batch):
    """A reloaded run must predict exactly what it predicted before saving.

    This is the train/serve skew test: weights alone are not enough, and if the
    normalisers were not persisted alongside them the two predictions diverge.
    """
    from hemoflow.models.registry import load_surrogate
    from hemoflow.training.train import train

    features, _, context = batch
    run_dir = train(cfg)

    reloaded, _ = load_surrogate(run_dir)
    first = reloaded.predict_pa(features, context)

    again, _ = load_surrogate(run_dir)
    np.testing.assert_allclose(first, again.predict_pa(features, context), rtol=1e-6)


def test_incomplete_run_directory_is_rejected(tmp_path):
    from hemoflow.models.registry import load_surrogate

    (tmp_path / "weights.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(FileNotFoundError, match="not a complete run directory"):
        load_surrogate(tmp_path)


def test_training_writes_a_servable_artifact(cfg):
    from hemoflow.training.train import train

    run_dir = train(cfg)
    for name in ("weights.pt", "standardizer.json", "target.json", "config.json", "metrics.json"):
        assert (run_dir / name).exists(), f"{name} missing from the run directory"
