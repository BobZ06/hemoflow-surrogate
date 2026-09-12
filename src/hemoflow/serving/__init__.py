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

RUN_DIR_ENV = "HEMOFLOW_RUN_DIR"
"""Pins the checkpoint the service and the demo load, instead of "newest run"."""

ABLATION_RUN_DIR_ENV = "HEMOFLOW_ABLATION_RUN_DIR"
"""Pins the no-physics-prior checkpoint behind the demo's ablation switch.

Both names live here rather than in `api` or `demo` because both modules read
them and neither may import the other: `api` mounts `demo`, so a reference the
other way would close the cycle.
"""

__all__ = [
    "ABLATION_RUN_DIR_ENV",
    "CFD_REFERENCE_SECONDS",
    "RUN_DIR_ENV",
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
