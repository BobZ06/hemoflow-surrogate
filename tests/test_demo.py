"""Properties the interactive demo must hold, not that its endpoints return 200.

The demo is the part of this repo a stranger judges it by, and its two headline
claims - "this is fast" and "the physics prior is what makes it right" - are
both claims a refactor could silently falsify while every page still rendered.
Both are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemoflow.config import Config, DataConfig, GeometryConfig, ModelConfig, PathConfig, TrainConfig

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A tiny trained run, pinned so the demo cannot pick up a stale checkpoint."""
    import os

    from hemoflow.serving import ABLATION_RUN_DIR_ENV, RUN_DIR_ENV
    from hemoflow.serving import api as api_module
    from hemoflow.serving import demo as demo_module
    from hemoflow.training.train import train

    tmp = tmp_path_factory.mktemp("demo")

    def build(name: str, *, physics_residual: bool) -> str:
        cfg = Config(
            name=name,
            geometry=GeometryConfig(n_arc=32, n_theta=16),
            data=DataConfig(n_geometries=64, seed=5, root=str(tmp / "data")),
            model=ModelConfig(name="mlp", hidden_dims=(64, 64), physics_residual=physics_residual),
            train=TrainConfig(epochs=40, batch_size=8, lr=3e-3, patience=10),
            paths=PathConfig(runs=str(tmp / "runs"), reports=str(tmp / "reports")),
        )
        return str(train(cfg))

    os.environ[RUN_DIR_ENV] = build("demo-prior", physics_residual=True)
    os.environ[ABLATION_RUN_DIR_ENV] = build("demo-ablation", physics_residual=False)
    demo_module._models.clear()

    yield TestClient(api_module.app)

    demo_module._models.clear()
    os.environ.pop(RUN_DIR_ENV, None)
    os.environ.pop(ABLATION_RUN_DIR_ENV, None)
    api_module._state.update({"surrogate": None, "config": None, "run_dir": None})


def _post(client, **overrides):
    body = {
        "stenosis_pct": 60.0,
        "stenosis_position": 0.6,
        "stenosis_width": 0.1,
        "inlet_diameter_mm": 4.0,
        "length_mm": 60.0,
        "curvature_per_m": 8.0,
        "flow_ml_s": 4.0,
        "physics_prior": True,
    }
    body.update(overrides)
    response = client.post("/demo/predict", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# The SVG namespace is a URI that identifies a namespace; no user agent ever
# fetches it. It is the one absolute URL allowed to appear in the page, and it
# is spelled out here so that adding a second one has to be a deliberate act.
INERT_URLS = ("http://www.w3.org/2000/svg",)


def test_page_is_served_and_carries_no_external_requests(client):
    """Hackathon wifi fails. Every byte the page needs must already be in it."""
    html = client.get("/").text
    assert "<canvas" in html
    for inert in INERT_URLS:
        html = html.replace(inert, "")
    for offender in ("http://", "https://", "//cdn", "integrity=", "@import"):
        assert offender not in html, f"page reaches off-box via {offender!r}"


def test_status_advertises_the_ablation_switch(client):
    status = client.get("/demo/status").json()
    assert status["ready"] is True
    assert status["ablation_available"] is True
    assert status["model"]["n_parameters"] > 0


def test_fields_share_one_grid(client):
    payload = _post(client)
    n_arc, n_theta = payload["grid"]
    for name, field in payload["fields"].items():
        array = np.asarray(field)
        assert array.shape == (n_arc, n_theta), f"{name} is not on the declared grid"
    assert len(payload["walls"]["outer"]) == n_arc
    assert len(payload["walls"]["inner"]) == n_arc


def test_wall_traces_lie_on_the_lumen_surface(client):
    """The drawn walls must be the vessel, not a decorative curve near it.

    The two traces are picked from discrete circumferential stations, so each
    sits at most half a station away from the section plane and the drawn
    half-width is `r * cos(delta)` rather than `r`. The bound below is that
    geometry exactly; anything outside it means the traces are not on the lumen.
    """
    payload = _post(client, curvature_per_m=0.0)
    n_theta = payload["grid"][1]
    radii = np.asarray(payload["radii_mm"])
    outer = np.asarray([p["x"] for p in payload["walls"]["outer"]])
    inner = np.asarray([p["x"] for p in payload["walls"]["inner"]])
    half_width = (outer - inner) / 2.0
    floor = radii * np.cos(np.pi / n_theta)
    assert np.all(half_width <= radii + 1e-3)
    assert np.all(half_width >= floor - 1e-3)


def test_the_throat_is_where_the_slider_put_it(client):
    """A control that moves the lesion somewhere else is worse than no control."""
    for position in (0.25, 0.5, 0.75):
        payload = _post(client, stenosis_position=position)
        radii = np.asarray(payload["radii_mm"])
        found = int(np.argmin(radii)) / (len(radii) - 1)
        assert abs(found - position) < 0.03


def test_speedup_is_consistent_with_the_timing_it_quotes(client):
    """The headline ratio must be derivable from the same response, so it cannot
    drift away from the measurement or the assumption it rests on."""
    payload = _post(client)
    timing = payload["timing_ms"]
    assert timing["query"] == pytest.approx(timing["features"] + timing["inference"], rel=1e-6)
    expected = payload["cfd_reference_seconds_assumed"] * 1000.0 / timing["query"]
    assert payload["speedup_vs_assumed_cfd"] == pytest.approx(expected, rel=0.01)


def test_physics_prior_keeps_absolute_scale_four_decades_out(client):
    """The demo's central technical claim.

    Wall shear stress scales as `1/r^3` at fixed flow. Nothing near capillary
    calibre appears in training, so a network emitting pascals directly can only
    return its training calibre - which is exactly what the ablation does, and
    exactly what the prior prevents. Halving the diameter at fixed flow must
    raise peak shear about eightfold.
    """
    large = _post(client, inlet_diameter_mm=4.0, flow_ml_s=4.0, stenosis_pct=0.0)
    small = _post(client, inlet_diameter_mm=2.0, flow_ml_s=4.0, stenosis_pct=0.0)
    ratio = small["summary"]["peak_pa"] / large["summary"]["peak_pa"]
    assert 6.0 < ratio < 10.5, f"scale is not carried by physics: ratio {ratio:.2f}"


def test_ablation_loses_that_scale(client):
    """The counterpart. Without the prior the same sweep must visibly fail to
    track `1/r^3`, otherwise the switch on the page is demonstrating nothing."""
    large = _post(
        client, inlet_diameter_mm=4.0, flow_ml_s=4.0, stenosis_pct=0.0, physics_prior=False
    )
    small = _post(
        client, inlet_diameter_mm=2.0, flow_ml_s=4.0, stenosis_pct=0.0, physics_prior=False
    )
    ratio = small["summary"]["peak_pa"] / large["summary"]["peak_pa"]
    assert ratio < 6.0, f"ablation unexpectedly recovered scale: ratio {ratio:.2f}"


def test_implausible_geometry_is_rejected_rather_than_answered(client):
    """A service that answers nonsense confidently is the failure mode that
    matters most in this domain."""
    response = client.post("/demo/predict", json={"stenosis_pct": 300.0})
    assert response.status_code == 422
