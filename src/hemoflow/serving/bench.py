"""Inference latency benchmark.

The entire argument for a surrogate is a latency claim, so the latency claim
needs to be measured rather than asserted, and measured the way it will actually
be paid: one vessel at a time, end to end, including feature extraction and
denormalisation, not just the forward pass.

Batched throughput is reported separately because it answers a different
question - how fast a cohort of a thousand scans can be processed offline - and
quoting a batched number as if it were interactive latency is the standard way
these claims get inflated.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..geometry import extract_features
from ..models.base import Surrogate, VesselContext
from ..utils import write_json

CFD_REFERENCE_SECONDS = 4.0 * 3600.0
"""Assumed cost of the 3D CFD solve a surrogate replaces: one case, one core-hour
budget of roughly four hours. This is an *assumption used for the speedup ratio*,
taken from typical published patient-specific runs - it is not measured here, and
any speedup figure quoting it must say so."""


def measure_latency(
    model: Surrogate,
    features: np.ndarray,
    context: VesselContext,
    repeats: int = 50,
    warmup: int = 5,
) -> dict[str, float]:
    """Time single-vessel inference.

    Args:
        model: any surrogate.
        features: `(n_vessels, n_arc, n_theta, n_features)` pool to sample from.
        context: matching physical context.
        repeats: timed iterations.
        warmup: untimed iterations first - the first call pays for lazy kernel
            selection and allocator warm-up, and including it inflates p50 on
            short benchmarks.

    Returns:
        Latency statistics in milliseconds plus single-case throughput.
    """
    n = len(features)
    single_context = VesselContext(
        flow_rate=context.flow_rate[:1],
        inlet_radius=context.inlet_radius[:1],
        viscosity=context.viscosity,
    )

    for i in range(warmup):
        model.predict_pa(features[i % n : i % n + 1], single_context)

    samples: list[float] = []
    for i in range(repeats):
        index = i % n
        ctx = VesselContext(
            flow_rate=context.flow_rate[index : index + 1],
            inlet_radius=context.inlet_radius[index : index + 1],
            viscosity=context.viscosity,
        )
        start = time.perf_counter()
        model.predict_pa(features[index : index + 1], ctx)
        samples.append((time.perf_counter() - start) * 1000.0)

    array = np.asarray(samples)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "cases_per_second": float(1000.0 / max(float(np.percentile(array, 50)), 1e-9)),
    }


def measure_feature_cost(cfg: Config, vessel: Any, repeats: int = 20) -> float:
    """Time feature extraction alone, in milliseconds.

    Reported because in a deployed system this is part of query latency, and on
    the smaller models it is the larger half of it.
    """
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        extract_features(vessel, cfg.fluid)
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples))


def measure_batch_throughput(
    model: Surrogate, features: np.ndarray, context: VesselContext, repeats: int = 5
) -> dict[str, float]:
    """Time a full-batch pass: the offline cohort-processing number."""
    for _ in range(2):
        model.predict_pa(features, context)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_pa(features, context)
        samples.append(time.perf_counter() - start)
    median = float(np.median(samples))
    return {
        "batch_size": int(len(features)),
        "batch_seconds": median,
        "vessels_per_second": float(len(features) / max(median, 1e-9)),
    }


def run_benchmark(
    cfg: Config,
    models: list[Surrogate],
    features: np.ndarray,
    context: VesselContext,
    cfd_reference_seconds: float = CFD_REFERENCE_SECONDS,
) -> dict[str, Any]:
    """Benchmark every model and write `reports/latency.json`."""
    rows: list[dict[str, Any]] = []
    for model in models:
        latency = measure_latency(model, features, context)
        throughput = measure_batch_throughput(model, features, context)
        speedup = cfd_reference_seconds / max(latency["p50_ms"] / 1000.0, 1e-12)
        rows.append(
            {
                "model": model.name,
                "params": model.n_parameters,
                **latency,
                **throughput,
                "speedup_vs_assumed_cfd": speedup,
            }
        )

    payload = {
        "cfd_reference_seconds": cfd_reference_seconds,
        "cfd_reference_is_an_assumption": True,
        "grid": [cfg.geometry.n_arc, cfg.geometry.n_theta],
        "rows": rows,
    }
    write_json(Path(cfg.paths.reports) / "latency.json", payload)
    return payload
