"""Torch dataset wrapper over the generated arrays.

Every model in the zoo consumes the same `(n_arc, n_theta, n_features)` grid.
Per-node models flatten it internally and grid models keep it; the dataset does
not need to know which is which, so swapping architectures never touches the
data path.

When `physics_residual` is on, the regression target is the dimensionless ratio
`tau_true / tau_poiseuille` rather than pascals - see `models/physics.py` for why
that is the right target for dimensionless features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..models.base import VesselContext
from ..models.physics import poiseuille_field
from .generate import load_context, load_split
from .normalize import Standardizer, TargetTransform


class VesselFieldDataset(Dataset):
    """One item per vessel: normalised features and the normalised target field.

    Args:
        features: `(n_vessels, n_arc, n_theta, n_features)` raw features.
        targets: `(n_vessels, n_arc, n_theta)` wall shear stress in pascals.
        standardizer: fitted on the train split only.
        target_transform: fitted on the train split only.
        prior: `(n_vessels, n_arc, n_theta)` physics prior in pascals, or None to
            regress absolute pascals directly.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        standardizer: Standardizer,
        target_transform: TargetTransform,
        prior: np.ndarray | None = None,
    ) -> None:
        if len(features) != len(targets):
            raise ValueError("features and targets disagree on the number of vessels")
        self.features = standardizer.transform(features)
        self.targets_pa = np.asarray(targets, dtype=np.float32)
        regression_target = targets if prior is None else np.asarray(targets) / prior
        self.targets = target_transform.forward(regression_target)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.asarray(self.features[index])),
            torch.from_numpy(np.asarray(self.targets[index])),
            torch.from_numpy(np.asarray(self.targets_pa[index])),
        )


def build_dataloaders(
    dataset_dir: str | Path,
    batch_size: int,
    log_space: bool = True,
    num_workers: int = 0,
    physics_residual: bool = True,
    viscosity: float = 0.0035,
) -> tuple[dict[str, DataLoader], Standardizer, TargetTransform]:
    """Build train/val/test loaders with normalisers fitted on train only.

    Returns:
        `(loaders, standardizer, target_transform)`. The two normalisers are
        returned so the caller can persist them beside the checkpoint - without
        them the weights are unusable at inference time.
    """
    dataset_dir = Path(dataset_dir)
    splits = ("train", "val", "test")
    raw = {split: load_split(dataset_dir, split) for split in splits}

    priors: dict[str, np.ndarray | None] = {}
    for split in splits:
        if not physics_residual:
            priors[split] = None
            continue
        arrays = load_context(dataset_dir, split)
        context = VesselContext(
            flow_rate=arrays["flow_rate"],
            inlet_radius=arrays["inlet_radius"],
            viscosity=viscosity,
        )
        priors[split] = poiseuille_field(raw[split][0], context)

    train_features, train_targets, _ = raw["train"]
    standardizer = Standardizer.fit(train_features)
    train_prior = priors["train"]
    fit_target = train_targets if train_prior is None else train_targets / train_prior
    target_transform = TargetTransform.fit(fit_target, log_space=log_space)

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        features, targets, _ = raw[split]
        dataset = VesselFieldDataset(
            features, targets, standardizer, target_transform, prior=priors[split]
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders, standardizer, target_transform
