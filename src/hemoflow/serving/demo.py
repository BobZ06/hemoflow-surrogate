"""The interactive demo: one vessel, five sliders, three fields, live timing.

Separate from `api.py` on purpose. `/predict` is the production-shaped endpoint -
one vessel in, a summary out, no opinion about how anyone draws it. The demo
needs something different: every field the viewer is comparing, the geometry to
draw them on, and a timing breakdown, all in one round trip so a slider drag
costs one request instead of four.

Keeping that in its own module means the demo's needs never leak into the
service contract. Deleting this file removes the demo and changes nothing else.

Two honesty rules are enforced here rather than left to the page:

* The timing breakdown reports feature extraction separately from the forward
  pass and adds them up. Feature extraction is a real part of query cost and
  quoting the forward pass alone would overstate the speedup.
* `reference_ms` is the *reduced-order* solver in this repo, not a 3D CFD solve.
  It is returned under that name and the page labels it that way. The CFD figure
  the speedup is quoted against is an assumption, and it travels with the
  response as `cfd_reference_seconds_assumed` so it can never be quietly
  detached from the claim it supports.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..config import Config, PathConfig
from ..geometry import extract_features, reference_wss, vessel_from_profile
from ..geometry.vessel import Vessel
from ..models.base import Surrogate, VesselContext
from ..models.physics import poiseuille_field
from ..training.metrics import LOW_SHEAR_THRESHOLD_PA, dice, relative_l2
from . import ABLATION_RUN_DIR_ENV, RUN_DIR_ENV
from .bench import CFD_REFERENCE_SECONDS

# The speedup denominator is the benchmark harness's assumed CFD cost, imported
# rather than redeclared. Two constants that must agree is one constant too
# many: the README table and this page would drift the first time either moved.
CFD_REFERENCE_SECONDS_ASSUMED = CFD_REFERENCE_SECONDS

router = APIRouter(tags=["demo"])

_PAGE = Path(__file__).parent / "static" / "demo.html"

# Loaded lazily and kept: reloading a checkpoint per slider tick would make the
# demo's own latency number meaningless.
_models: dict[str, Any] = {}


class DemoRequest(BaseModel):
    """A vessel described the way a slider panel describes one."""

    stenosis_pct: float = Field(60.0, ge=0, le=85, description="Focal narrowing, % diameter.")
    stenosis_position: float = Field(
        0.6, ge=0.15, le=0.85, description="Arc fraction of the throat."
    )
    stenosis_width: float = Field(
        0.10, ge=0.05, le=0.20, description="Lesion length, arc fraction."
    )
    # The lower bound reaches capillary calibre deliberately. Nothing near it
    # appears in training, and that is the point: the physics prior is what
    # keeps absolute scale correct four decades away from the training band, so
    # the demo has to be able to go there.
    inlet_diameter_mm: float = Field(4.6, gt=0.001, le=30.0, description="Healthy lumen diameter.")
    length_mm: float = Field(60.0, gt=5.0, le=300.0, description="Centerline arc length.")
    curvature_per_m: float = Field(8.0, ge=0.0, le=40.0, description="Constant centerline bend.")
    flow_ml_s: float = Field(2.8, gt=0.0, le=50.0, description="Volumetric inflow.")
    physics_prior: bool = Field(True, description="False serves the ablation checkpoint.")


def _run_dirs() -> list[Path]:
    runs = Path(PathConfig().runs)
    if not runs.exists():
        return []
    return [p for p in sorted(runs.iterdir()) if (p / "weights.pt").exists()]


def _pick_run(*, physics_residual: bool) -> Path | None:
    """Newest run whose config matches the requested physics-prior setting.

    The ablation checkpoint can also be pinned explicitly, because "newest run
    that happens to have the prior off" is a fragile thing for a live demo to
    depend on.
    """
    from ..config import load_config

    pinned = os.environ.get(ABLATION_RUN_DIR_ENV if not physics_residual else RUN_DIR_ENV)
    if pinned:
        return Path(pinned)

    matches = []
    for run_dir in _run_dirs():
        try:
            cfg = load_config(run_dir / "config.json")
        except Exception:  # noqa: BLE001 - a malformed run must not break the demo
            continue
        if cfg.model.physics_residual is physics_residual:
            matches.append(run_dir)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _load(*, physics_residual: bool) -> tuple[Surrogate, Config] | None:
    key = "with_prior" if physics_residual else "without_prior"
    if key in _models:
        return _models[key]

    run_dir = _pick_run(physics_residual=physics_residual)
    if run_dir is None or not (run_dir / "weights.pt").exists():
        _models[key] = None
        return None

    from ..models.registry import load_surrogate

    surrogate, cfg = load_surrogate(run_dir)
    _warm_up(surrogate, cfg)
    _models[key] = (surrogate, cfg, run_dir)
    return _models[key]


def _warm_up(surrogate: Surrogate, cfg: Config) -> None:
    """Run one throwaway prediction before the first real one.

    The first forward pass through a freshly loaded torch module pays for lazy
    kernel selection and allocator warm-up - tens of milliseconds that have
    nothing to do with steady-state cost. Letting the page's first request
    absorb that would put a number on screen that the benchmark contradicts.
    """
    n_arc, n_theta = cfg.geometry.n_arc, cfg.geometry.n_theta
    radii = np.full(n_arc, 2.0e-3)
    vessel = vessel_from_profile(
        radii=radii,
        length=0.06,
        curvature=0.0,
        n_theta=n_theta,
        flow_rate=4e-6,
        vessel_id="warmup",
    )
    features = extract_features(vessel, cfg.fluid)[None, ...]
    context = VesselContext(
        flow_rate=np.array([4e-6]),
        inlet_radius=np.array([2.0e-3]),
        viscosity=cfg.fluid.viscosity_pa_s,
    )
    surrogate.predict_pa(features, context)


def _radius_profile(request: DemoRequest, n_arc: int) -> np.ndarray:
    """Radius in metres along the vessel: a healthy lumen with one focal lesion.

    The lesion is the same Gaussian throat the training sampler draws, so a
    slider position is an in-distribution vessel rather than a shape the model
    has never seen. Severity is quoted by *diameter*, which is how stenosis is
    reported clinically.
    """
    s = np.linspace(0.0, 1.0, n_arc)
    base = request.inlet_diameter_mm / 2000.0  # mm diameter -> m radius
    severity = request.stenosis_pct / 100.0
    bump = np.exp(-0.5 * ((s - request.stenosis_position) / request.stenosis_width) ** 2)
    radius = base * (1.0 - severity * bump)
    return np.maximum(radius, 0.1 * base)


def _wall_traces(vessel: Vessel, field: np.ndarray) -> dict[str, list]:
    """Two wall polylines in the bend plane, with the field sampled on each.

    The centerline built by `vessel_from_profile` is planar in x-z, so a
    longitudinal section through that plane is the view a clinician would
    recognise. For every station the two drawn points are the ones whose radial
    direction is most nearly +x and -x: the outer and inner wall of the bend.
    Sampling the field there is what makes the Dean asymmetry visible - higher
    shear on the outer wall - rather than something you have to take on trust.
    """
    points = vessel.surface_points()  # (n_arc, n_theta, 3)
    radial = points - vessel.centerline[:, None, :]
    x_component = radial[:, :, 0]
    outer = np.argmax(x_component, axis=1)
    inner = np.argmin(x_component, axis=1)
    rows = np.arange(vessel.n_arc)

    def trace(index: np.ndarray) -> list[dict[str, float]]:
        picked = points[rows, index]
        values = field[rows, index]
        return [
            {"z": float(p[2] * 1000.0), "x": float(p[0] * 1000.0), "wss": float(v)}
            for p, v in zip(picked, values, strict=True)
        ]

    return {"outer": trace(outer), "inner": trace(inner)}


def _summary(field: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    low = field < LOW_SHEAR_THRESHOLD_PA
    low_reference = reference < LOW_SHEAR_THRESHOLD_PA
    return {
        "peak_pa": float(field.max()),
        "mean_pa": float(field.mean()),
        "min_pa": float(field.min()),
        "low_shear_fraction": float(low.mean()),
        "peak_arc_fraction": float(int(np.argmax(field.max(axis=1))) / max(len(field) - 1, 1)),
        "relative_l2_vs_reference": relative_l2(field, reference),
        "low_shear_dice_vs_reference": dice(low, low_reference),
    }


def _grid(field: np.ndarray) -> list[list[float]]:
    """Round before serialising. Three decimals of a pascal is far below the
    label noise and cuts the payload roughly in half."""
    return np.round(field, 3).tolist()


def _training_envelope(cfg: Config) -> dict[str, list[float]]:
    """The range of vessels the checkpoint was actually trained on.

    The page shades this onto the slider tracks. A surrogate reports no
    uncertainty - `docs/ROADMAP.md` calls that the gap that matters most - and
    until it does, the least a demo owes a viewer is to say plainly when they
    have dragged a control outside the distribution the model learned from.
    Derived from the config rather than written down, so it cannot go stale.
    """
    geometry, fluid = cfg.geometry, cfg.fluid
    # `sample_vessel` draws the healthy calibre as base_radius * U(0.75, 1.25),
    # and the inlet is where the taper starts, so inlet diameter spans that band.
    base_mm = geometry.base_radius_m * 2000.0
    return {
        "stenosis_pct": [0.0, geometry.max_stenosis * 100.0],
        "inlet_diameter_mm": [base_mm * 0.75, base_mm * 1.25],
        "flow_ml_s": [fluid.flow_min_m3_s * 1e6, fluid.flow_max_m3_s * 1e6],
    }


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> HTMLResponse:
    if not _PAGE.exists():  # pragma: no cover - only if the install is broken
        raise HTTPException(status_code=500, detail=f"demo page missing at {_PAGE}")
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"))


@router.get("/demo/status")
def demo_status() -> dict[str, Any]:
    """What the page needs to know before it renders a single control."""
    from ..utils import read_json

    loaded = _load(physics_residual=True)
    ablation = _load(physics_residual=False)

    if loaded is None:
        return {
            "ready": False,
            "detail": "No trained run found. Run `make train` first.",
            "ablation_available": False,
        }

    surrogate, cfg, run_dir = loaded
    meta_path = run_dir / "meta.json"
    metrics_path = run_dir / "metrics.json"
    return {
        "ready": True,
        "model": {
            "name": surrogate.name,
            "n_parameters": surrogate.n_parameters,
            "run_id": cfg.run_id,
            "grid": [cfg.geometry.n_arc, cfg.geometry.n_theta],
        },
        "meta": read_json(meta_path) if meta_path.exists() else None,
        "test_metrics": read_json(metrics_path) if metrics_path.exists() else None,
        "ablation_available": ablation is not None,
        "training_envelope": _training_envelope(cfg),
        "low_shear_threshold_pa": LOW_SHEAR_THRESHOLD_PA,
        "cfd_reference_seconds_assumed": CFD_REFERENCE_SECONDS_ASSUMED,
    }


@router.post("/demo/predict")
def demo_predict(request: DemoRequest) -> dict[str, Any]:
    """One vessel, every field the page compares, and an honest timing split."""
    started = time.perf_counter()

    loaded = _load(physics_residual=request.physics_prior)
    if loaded is None:
        detail = (
            "no ablation checkpoint; train configs/mlp_no_physics.yaml"
            if not request.physics_prior
            else "no trained run found; run `make train`"
        )
        raise HTTPException(status_code=503, detail=detail)
    surrogate, cfg, _ = loaded

    radii = _radius_profile(request, cfg.geometry.n_arc)
    flow_m3_s = request.flow_ml_s * 1e-6

    try:
        vessel = vessel_from_profile(
            radii=radii,
            length=request.length_mm / 1000.0,
            curvature=request.curvature_per_m,
            n_theta=cfg.geometry.n_theta,
            flow_rate=flow_m3_s,
            vessel_id="demo",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    feature_start = time.perf_counter()
    features = extract_features(vessel, cfg.fluid)[None, ...]
    features_ms = (time.perf_counter() - feature_start) * 1000.0

    context = VesselContext(
        flow_rate=np.array([flow_m3_s]),
        inlet_radius=np.array([float(radii[0])]),
        viscosity=cfg.fluid.viscosity_pa_s,
    )

    inference_start = time.perf_counter()
    predicted = np.asarray(surrogate.predict_pa(features, context))[0]
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    reference_start = time.perf_counter()
    reference = reference_wss(vessel, cfg.fluid)
    reference_ms = (time.perf_counter() - reference_start) * 1000.0

    baseline = poiseuille_field(features, context)[0]

    # Round the parts first, then sum them. A response whose timing breakdown
    # does not add up to its own total invites exactly the suspicion the
    # breakdown exists to remove.
    features_ms = round(features_ms, 3)
    inference_ms = round(inference_ms, 3)
    query_ms = round(features_ms + inference_ms, 3)
    return {
        "grid": [vessel.n_arc, vessel.n_theta],
        "length_mm": request.length_mm,
        "radii_mm": np.round(radii * 1000.0, 4).tolist(),
        "throat_diameter_mm": float(round(radii.min() * 2000.0, 4)),
        "fields": {
            "surrogate": _grid(predicted),
            "reference": _grid(reference),
            "baseline": _grid(baseline),
        },
        "walls": _wall_traces(vessel, predicted),
        "summary": _summary(predicted, reference),
        "reference_summary": _summary(reference, reference),
        "baseline_summary": _summary(baseline, reference),
        "timing_ms": {
            "features": features_ms,
            "inference": inference_ms,
            "query": query_ms,
            "reduced_order_reference": round(reference_ms, 3),
            "total_request": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "speedup_vs_assumed_cfd": round(
            CFD_REFERENCE_SECONDS_ASSUMED * 1000.0 / max(query_ms, 1e-9)
        ),
        "cfd_reference_seconds_assumed": CFD_REFERENCE_SECONDS_ASSUMED,
        "physics_prior": request.physics_prior,
        "model_name": surrogate.name,
    }
