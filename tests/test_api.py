"""Service contract: validation, unit handling, and physical sanity of responses."""

from __future__ import annotations

import numpy as np
import pytest

from hemoflow.config import Config, DataConfig, GeometryConfig, ModelConfig, PathConfig, TrainConfig

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Train a tiny model, point the service at it, and hand back a test client."""
    from hemoflow.serving import api as api_module
    from hemoflow.training.train import train

    tmp = tmp_path_factory.mktemp("service")
    cfg = Config(
        name="api",
        geometry=GeometryConfig(n_arc=32, n_theta=16),
        data=DataConfig(n_geometries=64, seed=11, root=str(tmp / "data")),
        model=ModelConfig(name="mlp", hidden_dims=(64, 64)),
        train=TrainConfig(epochs=40, batch_size=8, lr=3e-3, patience=10),
        paths=PathConfig(runs=str(tmp / "runs"), reports=str(tmp / "reports")),
    )
    run_dir = train(cfg)

    api_module._state.update({"surrogate": None, "config": None, "run_dir": None})
    import os

    os.environ[api_module.RUN_DIR_ENV] = str(run_dir)
    yield TestClient(api_module.app)
    os.environ.pop(api_module.RUN_DIR_ENV, None)
    api_module._state.update({"surrogate": None, "config": None, "run_dir": None})


def _body(**overrides):
    radii = (2.5 * (1.0 - 0.4 * np.exp(-0.5 * ((np.linspace(0, 1, 40) - 0.5) / 0.1) ** 2))).tolist()
    payload = {"radii_mm": radii, "length_mm": 60.0, "curvature_per_m": 20.0, "flow_ml_s": 4.0}
    payload.update(overrides)
    return payload


def test_health_reports_the_loaded_artifact(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["meta"]["architecture"] == "mlp"
    assert payload["meta"]["n_parameters"] > 0


def test_metrics_endpoint_returns_recorded_scores(client):
    payload = client.get("/metrics").json()
    assert "relative_l2" in payload
    assert payload["n_vessels"] > 0


def test_predict_returns_a_physically_plausible_field(client):
    payload = client.post("/predict", json=_body()).json()
    assert payload["peak_wss_pa"] > payload["mean_wss_pa"] > payload["min_wss_pa"] > 0
    assert 0.0 <= payload["low_shear_area_fraction"] <= 1.0
    assert 0.0 <= payload["peak_location_arc_fraction"] <= 1.0
    assert payload["inference_ms"] > 0
    assert payload["field_pa"] is None, "the full field must be opt-in"


def test_predict_can_return_the_full_field(client):
    payload = client.post("/predict", json=_body(include_field=True)).json()
    field = np.asarray(payload["field_pa"])
    assert field.shape == tuple(payload["grid"])
    assert np.isfinite(field).all()


def test_peak_shear_lands_near_the_throat(client):
    """A physics smoke test, not an accuracy claim.

    The narrowest point of the supplied profile is at the midpoint, so the peak
    shear must be somewhere near it. If this drifts to an end, a coordinate or
    resampling bug has crept in.
    """
    payload = client.post("/predict", json=_body()).json()
    assert 0.3 < payload["peak_location_arc_fraction"] < 0.7


def test_tighter_stenosis_raises_peak_shear(client):
    """WSS scales as 1/r^3, so the service must be monotone in severity."""
    s = np.linspace(0, 1, 40)
    mild = (2.5 * (1.0 - 0.2 * np.exp(-0.5 * ((s - 0.5) / 0.1) ** 2))).tolist()
    severe = (2.5 * (1.0 - 0.6 * np.exp(-0.5 * ((s - 0.5) / 0.1) ** 2))).tolist()

    mild_peak = client.post("/predict", json=_body(radii_mm=mild)).json()["peak_wss_pa"]
    severe_peak = client.post("/predict", json=_body(radii_mm=severe)).json()["peak_wss_pa"]
    assert severe_peak > mild_peak


def test_higher_flow_raises_shear(client):
    low = client.post("/predict", json=_body(flow_ml_s=2.0)).json()["mean_wss_pa"]
    high = client.post("/predict", json=_body(flow_ml_s=8.0)).json()["mean_wss_pa"]
    assert high > low


def test_profile_of_any_length_is_resampled_onto_the_training_grid(client):
    for n in (8, 40, 200):
        radii = np.full(n, 2.5).tolist()
        payload = client.post("/predict", json=_body(radii_mm=radii)).json()
        assert payload["grid"] == [32, 16]


@pytest.mark.parametrize(
    "bad",
    [
        {"radii_mm": [2.5, 2.5]},  # too few stations to differentiate
        {"radii_mm": [2.5, 2.5, 0.0, 2.5]},  # non-positive radius
        {"flow_ml_s": -1.0},
        {"length_mm": 0.0},
    ],
)
def test_invalid_requests_are_rejected(client, bad):
    assert client.post("/predict", json=_body(**bad)).status_code == 422


def test_absolute_scale_follows_the_physics_prior(client):
    """Shear must scale as 1/r^3 across calibres the model never trained on.

    This is the test that caught the original design bug. Every feature is
    dimensionless, so a network emitting pascals directly cannot recover absolute
    scale and just learns its training calibre - it answered a plausible 2 Pa for
    a 2.5-micron vessel. With the physics prior (`models/physics.py`) the scale
    comes from the analytic solution and this holds at any size.
    """
    n = 40
    base = client.post("/predict", json=_body(radii_mm=np.full(n, 2.5).tolist())).json()
    half = client.post("/predict", json=_body(radii_mm=np.full(n, 1.25).tolist())).json()

    ratio = half["mean_wss_pa"] / base["mean_wss_pa"]
    assert 6.0 < ratio < 10.0, f"halving the radius should raise shear ~8x, got {ratio:.2f}x"

    tiny = client.post("/predict", json=_body(radii_mm=np.full(n, 0.0025).tolist())).json()
    assert tiny["peak_wss_pa"] > 1e6, "a micron-scale vessel must not return a plausible value"
