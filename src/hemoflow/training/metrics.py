"""Evaluation metrics, including ones that reflect what the field is used for.

Mean absolute error over every node is a bad primary metric for this problem.
The target spans two orders of magnitude, so node-averaged error is dominated by
a handful of high-shear nodes at stenosis throats, and a model can post an
excellent MAE while being useless in the low-shear regions - which are the ones
that matter, because persistently low and oscillating wall shear is what drives
atherosclerotic plaque.

So the summary carries three kinds of number:

* **Scale-free accuracy** - relative L2 and median relative error, which weigh a
  10% error at 0.3 Pa the same as at 30 Pa.
* **Regional accuracy** - the same errors restricted to the low-shear band, so a
  model cannot hide a failure there behind good throat predictions.
* **Decision agreement** - Dice overlap between the predicted and reference
  low-shear regions. This is the closest thing here to the question a clinician
  would actually ask, which is *where* is the at-risk territory, not what the
  pascal value is at node 4,177.
"""

from __future__ import annotations

import numpy as np

LOW_SHEAR_THRESHOLD_PA = 1.0
"""Below roughly 1 Pa, endothelium shifts to an atheroprone phenotype. The exact
number is convention-dependent; it is a parameter everywhere it is used."""


def _flat(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64).reshape(-1)


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """`||pred - true||_2 / ||true||_2`, the standard surrogate-modelling metric."""
    p, t = _flat(pred), _flat(true)
    denom = float(np.linalg.norm(t))
    return float(np.linalg.norm(p - t) / denom) if denom > 0 else float("nan")


def dice(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Overlap of two boolean regions: `2|A and B| / (|A| + |B|)`."""
    p, t = np.asarray(pred_mask, bool), np.asarray(true_mask, bool)
    total = int(p.sum()) + int(t.sum())
    if total == 0:
        return 1.0  # both empty: perfect agreement that there is no low-shear region
    return float(2.0 * np.logical_and(p, t).sum() / total)


def compute_metrics(
    pred_pa: np.ndarray,
    true_pa: np.ndarray,
    low_shear_threshold: float = LOW_SHEAR_THRESHOLD_PA,
) -> dict[str, float]:
    """Score a batch of predicted WSS fields against the reference.

    Args:
        pred_pa: `(n_vessels, n_arc, n_theta)` predicted wall shear stress, Pa.
        true_pa: same shape, reference values, Pa.
        low_shear_threshold: boundary of the atheroprone band, Pa.

    Returns:
        Flat dict of scalar metrics, JSON-serialisable.
    """
    pred = np.asarray(pred_pa, dtype=np.float64)
    true = np.asarray(true_pa, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: predicted {pred.shape} vs reference {true.shape}")

    error = pred - true
    relative = np.abs(error) / np.maximum(np.abs(true), 1e-8)

    # Peak shear per vessel: the throat value a clinician would read off.
    peak_pred = pred.reshape(len(pred), -1).max(axis=1)
    peak_true = true.reshape(len(true), -1).max(axis=1)
    peak_relative = np.abs(peak_pred - peak_true) / np.maximum(peak_true, 1e-8)

    low_true = true < low_shear_threshold
    low_pred = pred < low_shear_threshold
    per_vessel_dice = [dice(low_pred[i], low_true[i]) for i in range(len(true))]

    metrics = {
        "mae_pa": float(np.abs(error).mean()),
        "rmse_pa": float(np.sqrt(np.mean(error**2))),
        "relative_l2": relative_l2(pred, true),
        "median_relative_error": float(np.median(relative)),
        "p90_relative_error": float(np.percentile(relative, 90)),
        "r2": float(1.0 - np.sum(error**2) / max(np.sum((true - true.mean()) ** 2), 1e-12)),
        "peak_median_relative_error": float(np.median(peak_relative)),
        "low_shear_dice": float(np.mean(per_vessel_dice)),
        "low_shear_fraction_true": float(low_true.mean()),
        "n_vessels": int(len(true)),
    }

    # Accuracy restricted to the atheroprone band. Reported separately because a
    # model tuned on the global average will quietly be worst exactly here.
    if low_true.any():
        metrics["low_shear_mae_pa"] = float(np.abs(error[low_true]).mean())
        metrics["low_shear_median_relative_error"] = float(np.median(relative[low_true]))
    else:
        metrics["low_shear_mae_pa"] = float("nan")
        metrics["low_shear_median_relative_error"] = float("nan")

    return metrics


def format_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    """Render metric rows as a GitHub-flavoured markdown table."""
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, rule]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            cells.append(f"{value:.4g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
