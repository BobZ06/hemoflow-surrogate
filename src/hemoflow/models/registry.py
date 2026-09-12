"""Checkpointing and model loading.

A checkpoint here is not just weights. Weights alone are useless: without the
feature standardiser and the target transform that were fitted alongside them,
the same tensor produces different numbers at serving time than it did in
validation. So a run directory is a self-contained unit -

    run_dir/
      config.json        the exact config that produced this run
      weights.pt         state dict
      standardizer.json  feature scaling, fitted on train only
      target.json        target transform, fitted on train only
      metrics.json       what it scored, and on which dataset

- and `load_surrogate(run_dir)` reconstitutes all of it in one call. The
inference service takes a run directory and nothing else, which means it is
structurally incapable of the train/serve skew that comes from re-deriving
normalisation at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import Config, load_config
from ..data.normalize import Standardizer, TargetTransform
from ..geometry.features import FEATURE_NAMES
from ..utils import git_sha, utc_now, write_json
from .base import Surrogate, VesselContext
from .nets import build_network
from .physics import poiseuille_field


class TorchSurrogate(Surrogate):
    """Adapter that makes a torch module satisfy the `Surrogate` interface.

    Owns the normalisation on both ends, so callers pass raw features and get
    pascals back.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        standardizer: Standardizer,
        target_transform: TargetTransform,
        name: str = "torch",
        device: str = "cpu",
        physics_residual: bool = True,
    ) -> None:
        self.module = module.to(device).eval()
        self.standardizer = standardizer
        self.target_transform = target_transform
        self.name = name
        self.device = device
        self.physics_residual = physics_residual

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    @torch.inference_mode()
    def predict_pa(self, features: np.ndarray, context: VesselContext) -> np.ndarray:
        scaled = self.standardizer.transform(features)
        tensor = torch.from_numpy(np.ascontiguousarray(scaled)).to(self.device)
        if tensor.ndim == 3:  # a single vessel: add the batch axis
            tensor = tensor.unsqueeze(0)
        out = self.target_transform.inverse(self.module(tensor).cpu().numpy())
        if not self.physics_residual:
            return out
        # `out` is the dimensionless ratio tau_true / tau_poiseuille; absolute
        # scale comes from the analytic prior, which knows the real radius.
        return out * poiseuille_field(features, context)


def save_checkpoint(
    run_dir: str | Path,
    module: torch.nn.Module,
    cfg: Config,
    standardizer: Standardizer,
    target_transform: TargetTransform,
    metrics: dict[str, Any] | None = None,
    dataset_dir: str | Path | None = None,
) -> Path:
    """Write a complete, self-contained run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(module.state_dict(), run_dir / "weights.pt")
    standardizer.save(run_dir / "standardizer.json")
    target_transform.save(run_dir / "target.json")
    write_json(run_dir / "config.json", cfg)
    write_json(
        run_dir / "meta.json",
        {
            "run_id": cfg.run_id,
            "created_utc": utc_now(),
            "git_sha": git_sha(),
            "architecture": cfg.model.name,
            "n_parameters": sum(p.numel() for p in module.parameters()),
            "feature_names": list(FEATURE_NAMES),
            "dataset_dir": str(dataset_dir) if dataset_dir else None,
            "dataset_id": cfg.dataset_id,
        },
    )
    if metrics is not None:
        write_json(run_dir / "metrics.json", metrics)
    return run_dir


def load_surrogate(run_dir: str | Path, device: str = "cpu") -> tuple[TorchSurrogate, Config]:
    """Rebuild a trained surrogate from a run directory.

    Returns:
        `(surrogate, config)`. The config comes back too so the caller knows the
        grid shape the model expects.
    """
    run_dir = Path(run_dir)
    missing = [
        f
        for f in ("weights.pt", "standardizer.json", "target.json", "config.json")
        if not (run_dir / f).exists()
    ]
    if missing:
        raise FileNotFoundError(f"{run_dir} is not a complete run directory; missing {missing}")

    cfg = load_config(run_dir / "config.json")
    module = build_network(
        cfg.model.name,
        n_features=len(FEATURE_NAMES),
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        base_channels=cfg.model.base_channels,
        depth=cfg.model.depth,
    )
    state = torch.load(run_dir / "weights.pt", map_location=device, weights_only=True)
    module.load_state_dict(state)

    surrogate = TorchSurrogate(
        module=module,
        standardizer=Standardizer.load(run_dir / "standardizer.json"),
        target_transform=TargetTransform.load(run_dir / "target.json"),
        name=cfg.model.name,
        device=device,
        physics_residual=cfg.model.physics_residual,
    )
    return surrogate, cfg


def resolve_device(requested: str = "auto") -> str:
    """Pick a device, honouring an explicit request."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
