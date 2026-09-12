"""Feature and target normalisation, fitted on train only and persisted with the model.

Two failure modes this module exists to prevent:

1. **Statistics leakage.** Fitting a scaler on the full dataset before splitting
   leaks test-set information into training. The scaler here is fitted from the
   train split alone and then applied to val and test unchanged.
2. **Train/serve skew.** A scaler that lives only in a notebook cannot be
   reapplied at inference time, so the deployed model silently sees differently
   scaled inputs. `Standardizer` and `TargetTransform` both serialise to JSON and
   are written into the run directory next to the weights, and the service loads
   them from there rather than recomputing anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import read_json, write_json


@dataclass(frozen=True)
class Standardizer:
    """Per-channel zero-mean unit-variance scaling of the node features."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> Standardizer:
        """Fit over every node of every vessel in the *training* split.

        Args:
            features: `(n_vessels, n_arc, n_theta, n_features)`.
        """
        flat = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        # A constant channel has zero variance; dividing by it produces NaNs that
        # only surface as a silently dead model many epochs later.
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((np.asarray(features, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Standardizer:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
        )

    def save(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> Standardizer:
        return cls.from_dict(read_json(path))


@dataclass(frozen=True)
class TargetTransform:
    """Log-space target scaling.

    Wall shear stress spans two orders of magnitude across a dataset containing
    severe stenoses: a few tenths of a pascal in a recirculation zone, tens of
    pascals at a throat. Regressing on raw pascals under an L2 loss lets a
    handful of throat nodes dominate every gradient, and the model buys accuracy
    there by giving up on the low-shear regions - which are the clinically
    interesting ones, since low and oscillatory shear is what drives plaque.

    Training on `log(tau)` makes the loss scale-free: a 10% error costs the same
    at 0.3 Pa as at 30 Pa.
    """

    mean: float
    std: float
    log_space: bool = True

    @classmethod
    def fit(cls, targets: np.ndarray, log_space: bool = True) -> TargetTransform:
        values = np.asarray(targets, dtype=np.float64)
        if log_space:
            values = np.log(np.maximum(values, 1e-8))
        std = float(values.std())
        return cls(mean=float(values.mean()), std=std if std > 1e-8 else 1.0, log_space=log_space)

    def forward(self, targets: np.ndarray) -> np.ndarray:
        """Pascals -> normalised model space."""
        values = np.asarray(targets, dtype=np.float64)
        if self.log_space:
            values = np.log(np.maximum(values, 1e-8))
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Normalised model space -> pascals.

        Every metric and every API response goes through here, so reported
        numbers are always in physical units.
        """
        out = np.asarray(values, dtype=np.float64) * self.std + self.mean
        if self.log_space:
            out = np.exp(np.clip(out, -30.0, 30.0))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "log_space": self.log_space}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TargetTransform:
        return cls(
            mean=float(payload["mean"]),
            std=float(payload["std"]),
            log_space=bool(payload["log_space"]),
        )

    def save(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> TargetTransform:
        return cls.from_dict(read_json(path))
