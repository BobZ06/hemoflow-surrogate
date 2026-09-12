"""Typed, hashable experiment configuration.

Every run is fully described by one YAML file. Nothing is passed by flag, so a
result can always be reproduced from the config that is committed next to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from .utils import fingerprint


@dataclass(frozen=True)
class GeometryConfig:
    """Discretisation of a vessel surface into an (arc x circumference) grid."""

    n_arc: int = 128
    n_theta: int = 64
    length_m: float = 0.06
    base_radius_m: float = 0.0025
    radius_jitter: float = 0.15
    max_stenosis: float = 0.55
    curvature_scale: float = 0.0010


@dataclass(frozen=True)
class FluidConfig:
    """Blood modelled as a Newtonian fluid; adequate in large arteries."""

    viscosity_pa_s: float = 0.0035
    density_kg_m3: float = 1060.0
    flow_min_m3_s: float = 1.5e-6
    flow_max_m3_s: float = 8.0e-6


@dataclass(frozen=True)
class DataConfig:
    n_geometries: int = 512
    seed: int = 1337
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    noise_sigma: float = 0.02
    root: str = "artifacts/data"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and the physics-prior switch.

    `physics_residual` is the single most consequential flag here. With it on,
    the network predicts a dimensionless *correction* to the analytic Poiseuille
    solution rather than absolute pascals; see `models/registry.py`.
    """

    name: str = "mlp"
    hidden_dims: tuple[int, ...] = (128, 128, 128)
    dropout: float = 0.0
    base_channels: int = 32
    depth: int = 3
    physics_residual: bool = True


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-5
    patience: int = 8
    log_space: bool = True
    device: str = "auto"
    num_workers: int = 0


@dataclass(frozen=True)
class PathConfig:
    runs: str = "artifacts/runs"
    reports: str = "artifacts/reports"


@dataclass(frozen=True)
class Config:
    name: str = "default"
    seed: int = 1337
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    fluid: FluidConfig = field(default_factory=FluidConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # -- derived -------------------------------------------------------------
    @property
    def run_id(self) -> str:
        """Config hash. Same config in, same directory out."""
        return f"{self.name}-{fingerprint(self)}"

    @property
    def run_dir(self) -> Path:
        return Path(self.paths.runs) / self.run_id

    @property
    def dataset_id(self) -> str:
        """Hash of only the parts that affect the dataset.

        Changing the learning rate must not invalidate a generated dataset, so
        the dataset fingerprint deliberately excludes model and train blocks.
        """
        return fingerprint((self.geometry, self.fluid, self.data))

    @property
    def dataset_dir(self) -> Path:
        return Path(self.data.root) / self.dataset_id

    def validate(self) -> None:
        if self.data.val_fraction + self.data.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must leave room for training data")
        if self.data.n_geometries < 8:
            raise ValueError("need at least 8 geometries to form three non-empty splits")
        if self.geometry.max_stenosis >= 1.0:
            raise ValueError("max_stenosis must be < 1.0 (1.0 is a fully occluded vessel)")
        if self.model.name not in {"poiseuille", "mlp", "unet"}:
            raise ValueError(f"unknown model '{self.model.name}'")


def _coerce(cls: type, value: Any) -> Any:
    """Build a (possibly nested) dataclass from plain dict data.

    `from __future__ import annotations` turns every field annotation into a
    string, so the real types are resolved with get_type_hints rather than read
    off the field objects.
    """
    if not is_dataclass(cls):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(value).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(value) - known
    if unknown:
        raise ValueError(f"unknown key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, raw in value.items():
        ftype = hints.get(key)
        if is_dataclass(ftype):
            kwargs[key] = _coerce(ftype, raw)
        elif isinstance(raw, list):
            kwargs[key] = tuple(raw)
        else:
            kwargs[key] = raw
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Read a YAML config, apply defaults, and validate it before anything runs."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = _coerce(Config, raw)
    cfg.validate()
    return cfg
