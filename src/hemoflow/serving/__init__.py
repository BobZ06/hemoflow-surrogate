"""Inference service and latency benchmarking.

`api` pulls in FastAPI and is imported lazily, so `hemoflow bench` works on a
machine that has torch but no web stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .bench import (
    CFD_REFERENCE_SECONDS,
    measure_batch_throughput,
    measure_latency,
    run_benchmark,
)

if TYPE_CHECKING:  # pragma: no cover
    from .api import app

__all__ = [
    "CFD_REFERENCE_SECONDS",
    "app",
    "measure_batch_throughput",
    "measure_latency",
    "run_benchmark",
]


def __getattr__(name: str) -> Any:
    if name == "app":
        from .api import app as _app

        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
