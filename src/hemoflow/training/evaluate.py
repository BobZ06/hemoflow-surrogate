"""Benchmark harness: score every model on the same held-out split.

The output of this module is the table that goes in the README. It is the only
place in the repo authorised to make a claim about how well anything works, and
it does so by scoring baselines and networks through one interface, on one
split, with one set of metrics.

Baselines are fitted here rather than loaded, because a fitted baseline must see
the training split and nothing else - fitting a "baseline" on the test data is a
quiet way to make a neural model look better than it is.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..data import generate_dataset, load_context, load_split
from ..models.base import Surrogate, VesselContext
from ..models.baselines import ConstantBaseline, PoiseuilleBaseline, RidgeBaseline
from ..utils import write_json
from .metrics import compute_metrics, format_table

logger = logging.getLogger(__name__)

REPORT_COLUMNS = [
    "model",
    "params",
    "relative_l2",
    "median_relative_error",
    "mae_pa",
    "low_shear_median_relative_error",
    "low_shear_dice",
    "peak_median_relative_error",
]


def _context_for(cfg: Config, dataset_dir: Path, split: str) -> VesselContext:
    arrays = load_context(dataset_dir, split)
    return VesselContext(
        flow_rate=arrays["flow_rate"],
        inlet_radius=arrays["inlet_radius"],
        viscosity=cfg.fluid.viscosity_pa_s,
    )


def build_baselines(train_features: np.ndarray, train_targets: np.ndarray) -> list[Surrogate]:
    """Fit the non-neural baselines on the training split only."""
    return [
        ConstantBaseline.fit(train_targets),
        PoiseuilleBaseline(),
        RidgeBaseline.fit(train_features, train_targets),
    ]


def evaluate_models(
    cfg: Config,
    models: list[Surrogate],
    split: str = "test",
) -> list[dict[str, Any]]:
    """Score a list of surrogates on one split.

    Returns:
        One row per model, sorted best-first by relative L2 error.
    """
    dataset_dir = generate_dataset(cfg)
    features, targets, _ = load_split(dataset_dir, split)
    context = _context_for(cfg, dataset_dir, split)

    rows: list[dict[str, Any]] = []
    for model in models:
        predictions = model.predict_pa(features, context)
        row: dict[str, Any] = {"model": model.name, "params": model.n_parameters}
        row.update(compute_metrics(predictions, targets))
        rows.append(row)

    return sorted(rows, key=lambda r: r["relative_l2"])


def benchmark(cfg: Config, run_dirs: list[str | Path] | None = None, split: str = "test") -> Path:
    """Run the full comparison and write `benchmark.json` and `benchmark.md`.

    Args:
        cfg: config describing the dataset to evaluate on.
        run_dirs: trained run directories to include alongside the baselines.
        split: which split to score on.

    Returns:
        Path to the written markdown report.
    """
    dataset_dir = generate_dataset(cfg)
    train_features, train_targets, _ = load_split(dataset_dir, "train")

    models: list[Surrogate] = build_baselines(train_features, train_targets)

    skipped: list[str] = []
    for run_dir in run_dirs or []:
        from ..models.registry import load_surrogate

        surrogate, run_cfg = load_surrogate(run_dir)

        # A run trained on a different dataset cannot share a row with these
        # baselines: same column headings, different held-out vessels. Comparing
        # them anyway is the kind of table that looks fine and means nothing.
        if run_cfg.dataset_id != cfg.dataset_id:
            logger.warning(
                "skipping %s: trained on dataset %s, evaluating on %s",
                Path(run_dir).name,
                run_cfg.dataset_id,
                cfg.dataset_id,
            )
            skipped.append(Path(run_dir).name)
            continue

        label = f"{run_cfg.model.name}"
        if not run_cfg.model.physics_residual:
            label += " (no physics prior)"
        surrogate.name = label
        models.append(surrogate)

    rows = evaluate_models(cfg, models, split=split)

    reports_dir = Path(cfg.paths.reports)
    write_json(reports_dir / "benchmark.json", {"split": split, "rows": rows})

    table = format_table(rows, REPORT_COLUMNS)
    report_path = reports_dir / "benchmark.md"
    report_path.write_text(
        "# Benchmark\n\n"
        f"Dataset `{cfg.dataset_id}` - {cfg.data.n_geometries} synthetic vessels, "
        f"scored on the `{split}` split.\n\n"
        "Reference fields come from the reduced-order solver in "
        "`geometry/reference.py`, **not** a 3D CFD solve, so these numbers measure "
        "fidelity to that reference and nothing more.\n\n"
        f"{table}\n\n"
        "`relative_l2` is the primary metric. `low_shear_*` columns are restricted to "
        "nodes below 1 Pa, the atheroprone band.\n"
        + (
            f"\nSkipped (trained on a different dataset): {', '.join(skipped)}\n"
            if skipped
            else ""
        )
    )
    return report_path
