"""Training loop: config in, run directory out.

Deliberately boring. Every knob lives in the config, the best checkpoint is
selected on validation loss rather than the last epoch, and the normalisers are
written next to the weights so the artifact is servable without any further
context. There is nothing here a second person has to be told in person in order
to reproduce a result.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..config import Config
from ..data import build_dataloaders, generate_dataset, load_context, load_split
from ..geometry.features import FEATURE_NAMES
from ..models.base import VesselContext
from ..models.nets import build_network
from ..models.registry import TorchSurrogate, resolve_device, save_checkpoint
from ..utils import set_seed, write_json
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


def _run_epoch(
    module: nn.Module,
    loader: Any,
    device: str,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """One pass over a loader. Trains when an optimizer is given, else evaluates."""
    training = optimizer is not None
    module.train(training)
    loss_fn = nn.MSELoss()

    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for features, target, _ in loader:
            features = features.to(device)
            target = target.to(device)

            prediction = module(features)
            loss = loss_fn(prediction, target)

            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Log-space targets keep gradients tame, but a severe stenosis
                # early in an epoch can still spike them; clipping costs nothing.
                torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
                optimizer.step()

            total += float(loss.item()) * len(features)
            count += len(features)
    return total / max(count, 1)


def train(cfg: Config, force_data: bool = False) -> Path:
    """Train the model described by `cfg` and return its run directory.

    Args:
        cfg: full experiment config.
        force_data: regenerate the dataset even if it already exists.

    Returns:
        Path to the run directory containing weights, normalisers, config,
        training history and test metrics.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device)

    dataset_dir = generate_dataset(cfg, force=force_data)
    loaders, standardizer, target_transform = build_dataloaders(
        dataset_dir,
        batch_size=cfg.train.batch_size,
        log_space=cfg.train.log_space,
        num_workers=cfg.train.num_workers,
        physics_residual=cfg.model.physics_residual,
        viscosity=cfg.fluid.viscosity_pa_s,
    )

    module = build_network(
        cfg.model.name,
        n_features=len(FEATURE_NAMES),
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        base_channels=cfg.model.base_channels,
        depth=cfg.model.depth,
    ).to(device)

    optimizer = torch.optim.AdamW(
        module.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    logger.info(
        "training %s on %d vessels (%d train / %d val), device=%s, physics_residual=%s",
        cfg.model.name,
        cfg.data.n_geometries,
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        device,
        cfg.model.physics_residual,
    )

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = _run_epoch(module, loaders["train"], device, optimizer)
        val_loss = _run_epoch(module, loaders["val"], device, None)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < best_val - 1e-6
        logger.info(
            "epoch %3d/%d  train %.5f  val %.5f  %s",
            epoch,
            cfg.train.epochs,
            train_loss,
            val_loss,
            "*" if improved else f"(no gain for {epochs_without_improvement + 1})",
        )

        if improved:
            best_val = val_loss
            # Detached CPU copy: keeping GPU tensors alive across epochs is how
            # a long sweep runs out of memory at epoch 40.
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.train.patience:
                logger.info("early stop: no improvement in %d epochs", cfg.train.patience)
                break

    if best_state is not None:
        module.load_state_dict(best_state)

    wall_time = time.perf_counter() - started

    # --- score the selected checkpoint on the untouched test split -----------
    surrogate = TorchSurrogate(
        module=module,
        standardizer=standardizer,
        target_transform=target_transform,
        name=cfg.model.name,
        device=device,
        physics_residual=cfg.model.physics_residual,
    )
    test_features, test_targets, _ = load_split(dataset_dir, "test")
    context_arrays = load_context(dataset_dir, "test")
    context = VesselContext(
        flow_rate=context_arrays["flow_rate"],
        inlet_radius=context_arrays["inlet_radius"],
        viscosity=cfg.fluid.viscosity_pa_s,
    )
    predictions = surrogate.predict_pa(test_features, context)
    metrics = compute_metrics(predictions, test_targets)
    metrics.update(
        {
            "best_val_loss": best_val,
            "epochs_run": len(history),
            "train_wall_seconds": wall_time,
            "n_parameters": surrogate.n_parameters,
            "device": device,
        }
    )

    run_dir = cfg.run_dir
    save_checkpoint(
        run_dir,
        module=module,
        cfg=cfg,
        standardizer=standardizer,
        target_transform=target_transform,
        metrics=metrics,
        dataset_dir=dataset_dir,
    )
    write_json(run_dir / "history.json", history)
    return run_dir


def load_eval_batch(
    cfg: Config, split: str = "test"
) -> tuple[np.ndarray, np.ndarray, VesselContext]:
    """Load one split plus its physical context, ready for `predict_pa`."""
    dataset_dir = generate_dataset(cfg)
    features, targets, _ = load_split(dataset_dir, split)
    arrays = load_context(dataset_dir, split)
    context = VesselContext(
        flow_rate=arrays["flow_rate"],
        inlet_radius=arrays["inlet_radius"],
        viscosity=cfg.fluid.viscosity_pa_s,
    )
    return features, targets, context
