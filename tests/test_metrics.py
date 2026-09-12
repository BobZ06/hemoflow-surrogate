"""Metric behaviour, especially the properties that make the benchmark honest."""

from __future__ import annotations

import numpy as np
import pytest

from hemoflow.training.metrics import (
    LOW_SHEAR_THRESHOLD_PA,
    compute_metrics,
    dice,
    format_table,
    relative_l2,
)


@pytest.fixture
def field() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.exp(rng.normal(0.5, 0.8, size=(6, 16, 8)))


def test_perfect_prediction_scores_perfectly(field):
    metrics = compute_metrics(field, field)
    assert metrics["relative_l2"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["mae_pa"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-9)
    assert metrics["low_shear_dice"] == pytest.approx(1.0)


def test_relative_l2_is_scale_free(field):
    """Doubling both prediction and reference must not change the relative error."""
    assert relative_l2(1.3 * field, field) == pytest.approx(relative_l2(2.6 * field, 2.0 * field))


def test_worse_predictions_score_worse(field):
    rng = np.random.default_rng(1)
    mild = field * np.exp(rng.normal(0, 0.05, field.shape))
    severe = field * np.exp(rng.normal(0, 0.4, field.shape))
    assert (
        compute_metrics(mild, field)["relative_l2"] < compute_metrics(severe, field)["relative_l2"]
    )


def test_low_shear_metrics_expose_a_failure_the_global_average_hides():
    """The point of the regional metrics, made concrete.

    A model that is excellent at high shear and hopeless below 1 Pa can post a
    respectable global MAE, because the high-shear nodes carry the magnitude.
    The low-shear columns must not let that pass.
    """
    truth = np.concatenate(
        [np.full((2, 8, 4), 0.4), np.full((2, 8, 4), 20.0)], axis=1
    )  # half atheroprone, half throat
    prediction = truth.copy()
    prediction[:, :8, :] = 1.6  # 4x wrong exactly where it matters

    metrics = compute_metrics(prediction, truth, low_shear_threshold=LOW_SHEAR_THRESHOLD_PA)

    assert metrics["median_relative_error"] < 2.0, "global view looks tolerable"
    assert metrics["low_shear_median_relative_error"] > 2.0, "regional view must expose it"
    assert metrics["low_shear_dice"] < 0.2, "the at-risk region is missed entirely"


def test_dice_handles_the_empty_case():
    empty = np.zeros((4, 4), dtype=bool)
    assert dice(empty, empty) == 1.0
    assert dice(np.ones((4, 4), dtype=bool), empty) == 0.0
    assert dice(np.ones((4, 4), dtype=bool), np.ones((4, 4), dtype=bool)) == 1.0


def test_peak_error_is_measured_per_vessel(field):
    """Peak shear is a per-case reading, so its error must not be pooled."""
    prediction = field.copy()
    prediction[0] *= 2.0  # one bad vessel out of six
    metrics = compute_metrics(prediction, field)
    # The median over vessels should stay near zero despite one large miss.
    assert metrics["peak_median_relative_error"] < 0.05


def test_shape_mismatch_is_rejected(field):
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics(field[:, :4], field)


def test_metrics_are_json_serialisable(field):
    import json

    json.dumps(compute_metrics(field, field))


def test_format_table_renders_markdown():
    table = format_table(
        [{"model": "unet", "relative_l2": 0.0421}, {"model": "poiseuille", "relative_l2": 0.3}],
        ["model", "relative_l2"],
    )
    lines = table.splitlines()
    assert lines[0].startswith("| model |")
    assert set(lines[1].replace(" ", "")) <= {"|", "-"}
    assert "unet" in lines[2]
