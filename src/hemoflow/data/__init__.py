"""Dataset generation, splits, normalisation, and torch loaders.

The torch-dependent loader is imported lazily so that `hemoflow data` can
generate a dataset on a machine that has only numpy installed - useful when the
data step runs somewhere other than the training box.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .generate import SPLIT_NAMES, generate_dataset, load_context, load_split, make_splits
from .normalize import Standardizer, TargetTransform

if TYPE_CHECKING:  # pragma: no cover
    from .dataset import VesselFieldDataset, build_dataloaders

__all__ = [
    "SPLIT_NAMES",
    "Standardizer",
    "TargetTransform",
    "VesselFieldDataset",
    "build_dataloaders",
    "generate_dataset",
    "load_context",
    "load_split",
    "make_splits",
]


def __getattr__(name: str) -> Any:
    if name in {"VesselFieldDataset", "build_dataloaders"}:
        from . import dataset as _dataset

        return getattr(_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
