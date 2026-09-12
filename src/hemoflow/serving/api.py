"""HTTP inference service.

A trained checkpoint that only runs inside a notebook is not a deliverable. This
exposes one behind a typed endpoint with validated inputs, per-request timing,
and a health check that reports which run directory is actually loaded - so the
demo and any downstream caller talk to exactly the artifact the benchmark
scored.

Run it with `make serve`, then open `/docs` for the generated OpenAPI page.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..config import PathConfig
from ..geometry import extract_features, reference_wss, vessel_from_profile
from ..models.base import VesselContext
from ..training.metrics import LOW_SHEAR_THRESHOLD_PA
from ..utils import read_json

RUN_DIR_ENV = "HEMOFLOW_RUN_DIR"


class PredictRequest(BaseModel):
    """A vessel to evaluate.

    Units are explicit in every field name. Unit confusion between millimetres
    and metres is the single most common source of silently wrong answers in
    this domain, so the API speaks clinical units and converts once, here.
    """

    radii_mm: list[float] = Field(
        ..., min_length=4, description="Lumen radius at each station along the vessel, inlet first."
    )
    length_mm: float = Field(60.0, gt=0, description="Centerline arc length.")
    curvature_per_m: float = Field(
        0.0, ge=0, description="Constant centerline curvature; 0 is a straight vessel."
    )
    flow_ml_s: float = Field(4.0, gt=0, description="Volumetric inflow in millilitres per second.")
    include_field: bool = Field(
        False, description="Return the full field as well as the summary. Large."
    )

    @field_validator("radii_mm")
    @classmethod
    def _positive_radii(cls, value: list[float]) -> list[float]:
        if any(r <= 0 for r in value):
            raise ValueError("every radius must be positive")
        if max(value) / min(value) > 50:
            raise ValueError("radius profile spans an implausible range; check units (mm expected)")
        return value


class PredictResponse(BaseModel):
    vessel_id: str
    model_name: str
    grid: list[int]
    peak_wss_pa: float
    mean_wss_pa: float
    min_wss_pa: float
    low_shear_area_fraction: float
    peak_location_arc_fraction: float
    reference_peak_wss_pa: float | None = None
    inference_ms: float
    total_ms: float
    field_pa: list[list[float]] | None = None


app = FastAPI(
    title="hemoflow",
    version="0.1.0",
    description="Neural surrogate for vascular wall shear stress.",
)

_state: dict[str, Any] = {"surrogate": None, "config": None, "run_dir": None}


def _resolve_run_dir() -> Path:
    """Find the checkpoint to serve: env var if set, else the newest run."""
    explicit = os.environ.get(RUN_DIR_ENV)
    if explicit:
        return Path(explicit)
    runs = Path(PathConfig().runs)
    candidates = [p for p in runs.glob("*") if (p / "weights.pt").exists()] if runs.exists() else []
    if not candidates:
        raise FileNotFoundError(
            "no trained run found. Train one with `make train`, "
            f"or point {RUN_DIR_ENV} at a run directory."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _ensure_loaded() -> None:
    if _state["surrogate"] is not None:
        return
    from ..models.registry import load_surrogate

    run_dir = _resolve_run_dir()
    surrogate, cfg = load_surrogate(run_dir)
    _state.update({"surrogate": surrogate, "config": cfg, "run_dir": run_dir})


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus the identity of the loaded artifact."""
    try:
        _ensure_loaded()
    except FileNotFoundError as exc:
        return {"status": "no_model", "detail": str(exc)}
    run_dir = Path(_state["run_dir"])
    meta_path = run_dir / "meta.json"
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "meta": read_json(meta_path) if meta_path.exists() else None,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Test-split metrics recorded when this checkpoint was trained."""
    _ensure_loaded()
    path = Path(_state["run_dir"]) / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no metrics recorded for this run")
    return read_json(path)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict the wall-shear-stress field for one vessel."""
    started = time.perf_counter()
    _ensure_loaded()

    surrogate = _state["surrogate"]
    cfg = _state["config"]

    radii_m = np.asarray(request.radii_mm, dtype=np.float64) / 1000.0
    flow_m3_s = request.flow_ml_s * 1e-6

    # The network was trained on a fixed grid, so resample the caller's profile
    # onto it rather than rejecting any profile of a different length.
    target_n = cfg.geometry.n_arc
    source = np.linspace(0.0, 1.0, len(radii_m))
    resampled = np.interp(np.linspace(0.0, 1.0, target_n), source, radii_m)

    try:
        vessel = vessel_from_profile(
            radii=resampled,
            length=request.length_mm / 1000.0,
            curvature=request.curvature_per_m,
            n_theta=cfg.geometry.n_theta,
            flow_rate=flow_m3_s,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    features = extract_features(vessel, cfg.fluid)[None, ...]
    context = VesselContext(
        flow_rate=np.array([flow_m3_s]),
        inlet_radius=np.array([float(resampled[0])]),
        viscosity=cfg.fluid.viscosity_pa_s,
    )

    inference_start = time.perf_counter()
    field = np.asarray(surrogate.predict_pa(features, context))[0]
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    peak_index = int(np.argmax(field.max(axis=1)))
    reference = reference_wss(vessel, cfg.fluid)

    return PredictResponse(
        vessel_id=vessel.vessel_id,
        model_name=surrogate.name,
        grid=[vessel.n_arc, vessel.n_theta],
        peak_wss_pa=float(field.max()),
        mean_wss_pa=float(field.mean()),
        min_wss_pa=float(field.min()),
        low_shear_area_fraction=float((field < LOW_SHEAR_THRESHOLD_PA).mean()),
        peak_location_arc_fraction=float(peak_index / max(vessel.n_arc - 1, 1)),
        reference_peak_wss_pa=float(reference.max()),
        inference_ms=inference_ms,
        total_ms=(time.perf_counter() - started) * 1000.0,
        field_pa=field.tolist() if request.include_field else None,
    )
