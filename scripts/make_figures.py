#!/usr/bin/env python
"""Render qualitative figures: reference field, prediction, and error map.

Numbers in a table tell you how wrong a model is on average. They do not tell
you *where* it is wrong, and for a field prediction that is usually the more
useful question - a model with a respectable relative L2 can still be putting
all of its error in one band, which the unwrapped error map makes obvious at a
glance.

    python scripts/make_figures.py --config configs/unet.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hemoflow.config import Config, load_config
from hemoflow.data import generate_dataset, load_context, load_split
from hemoflow.models.base import VesselContext
from hemoflow.models.baselines import PoiseuilleBaseline


def _latest_run(cfg: Config) -> Path | None:
    runs = Path(cfg.paths.runs)
    if not runs.exists():
        return None
    candidates = [p for p in runs.iterdir() if (p / "weights.pt").exists()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--index", type=int, default=0, help="which test vessel to plot")
    parser.add_argument("--out", default=None, help="output directory")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(args.config) if args.config else Config()
    out_dir = Path(args.out) if args.out else Path(cfg.paths.reports)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = generate_dataset(cfg)
    features, targets, _ = load_split(dataset_dir, "test")
    arrays = load_context(dataset_dir, "test")
    context = VesselContext(
        flow_rate=arrays["flow_rate"],
        inlet_radius=arrays["inlet_radius"],
        viscosity=cfg.fluid.viscosity_pa_s,
    )

    models = [("poiseuille", PoiseuilleBaseline())]
    run_dir = _latest_run(cfg)
    if run_dir is not None:
        from hemoflow.models.registry import load_surrogate

        surrogate, run_cfg = load_surrogate(run_dir)
        models.append((f"{run_cfg.model.name}", surrogate))

    i = args.index
    truth = targets[i]
    fig, axes = plt.subplots(
        len(models) + 1, 2, figsize=(11, 3.1 * (len(models) + 1)), constrained_layout=True
    )
    if axes.ndim == 1:
        axes = axes[None, :]

    vmin, vmax = float(truth.min()), float(np.percentile(truth, 99))
    extent = [0, 360, 0, 1]

    def show(ax, field, title, **kwargs):
        image = ax.imshow(field.T, origin="lower", aspect="auto", extent=extent, **kwargs)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("circumference (deg)")
        ax.set_ylabel("arc position")
        return image

    # Row 0: the reference field and the radius profile that produced it.
    image = show(axes[0, 0], truth.T, "reference WSS (Pa)", vmin=vmin, vmax=vmax, cmap="inferno")
    fig.colorbar(image, ax=axes[0, 0])

    radius_mm = context.inlet_radius[i] * np.exp(features[i, :, 0, 0]) * 1000
    axes[0, 1].plot(np.linspace(0, 1, len(radius_mm)), radius_mm, lw=2)
    axes[0, 1].set_title("radius profile (mm)", fontsize=10)
    axes[0, 1].set_xlabel("arc position")
    axes[0, 1].set_ylabel("radius (mm)")
    axes[0, 1].grid(alpha=0.3)

    for row, (name, model) in enumerate(models, start=1):
        prediction = np.asarray(
            model.predict_pa(
                features[i : i + 1],
                VesselContext(
                    flow_rate=context.flow_rate[i : i + 1],
                    inlet_radius=context.inlet_radius[i : i + 1],
                    viscosity=context.viscosity,
                ),
            )
        )[0]

        image = show(
            axes[row, 0],
            prediction.T,
            f"{name}: prediction (Pa)",
            vmin=vmin,
            vmax=vmax,
            cmap="inferno",
        )
        fig.colorbar(image, ax=axes[row, 0])

        relative = (prediction - truth) / np.maximum(truth, 1e-8) * 100
        limit = float(np.percentile(np.abs(relative), 99))
        image = show(
            axes[row, 1],
            relative.T,
            f"{name}: relative error (%)  median |err| {np.median(np.abs(relative)):.1f}%",
            vmin=-limit,
            vmax=limit,
            cmap="RdBu_r",
        )
        fig.colorbar(image, ax=axes[row, 1])

    path = out_dir / "fields.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")

    # Training curves, when a run exists.
    if run_dir is not None and (run_dir / "history.json").exists():
        from hemoflow.utils import read_json

        history = read_json(run_dir / "history.json")
        fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train")
        ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="val")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE (normalised log space)")
        ax.set_title(f"{run_dir.name}", fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
        curve_path = out_dir / "training_curve.png"
        fig.savefig(curve_path, dpi=130)
        plt.close(fig)
        print(f"wrote {curve_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
