"""Training loop, benchmark harness, and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .metrics import LOW_SHEAR_THRESHOLD_PA, compute_metrics, dice, format_table, relative_l2

if TYPE_CHECKING:  # pragma: no cover
    from .evaluate import benchmark, build_baselines, evaluate_models
    from .train import train

_LAZY = {
    "benchmark": "evaluate",
    "build_baselines": "evaluate",
    "evaluate_models": "evaluate",
    "train": "train",
}

__all__ = [
    "LOW_SHEAR_THRESHOLD_PA",
    "benchmark",
    "build_baselines",
    "compute_metrics",
    "dice",
    "evaluate_models",
    "format_table",
    "relative_l2",
    "train",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)
