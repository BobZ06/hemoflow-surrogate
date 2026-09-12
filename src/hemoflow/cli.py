"""Command-line entry point.

Built on argparse rather than a CLI framework, because the package should stay
installable with nothing but numpy, scipy and pyyaml. Every subcommand takes a
config file and nothing that changes results - there are no hyperparameter flags
to forget to write down.

    hemoflow data   --config configs/mlp.yaml    generate the dataset
    hemoflow train  --config configs/mlp.yaml    train and score
    hemoflow eval   --config configs/mlp.yaml    benchmark table vs. baselines
    hemoflow bench  --config configs/mlp.yaml    inference latency
    hemoflow info   --config configs/mlp.yaml    resolved config and ids
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config, load_config
from .utils import to_plain


def _load(args: argparse.Namespace) -> Config:
    return load_config(args.config) if args.config else Config()


def _latest_runs(cfg: Config) -> list[Path]:
    runs = Path(cfg.paths.runs)
    if not runs.exists():
        return []
    return sorted(
        (p for p in runs.iterdir() if (p / "weights.pt").exists()),
        key=lambda p: p.stat().st_mtime,
    )


def cmd_data(args: argparse.Namespace) -> int:
    from .data import generate_dataset
    from .utils import read_json

    cfg = _load(args)
    out = generate_dataset(cfg, force=args.force)
    manifest = read_json(out / "manifest.json")
    print(f"dataset {cfg.dataset_id} -> {out}")
    print(f"  vessels      {manifest['split_sizes']}")
    print(f"  grid         {manifest['shapes']['targets'][1:]}")
    stats = manifest["target_stats_pa"]
    print(
        f"  WSS (Pa)     min {stats['min']:.3f}  p50 {stats['p50']:.3f}  "
        f"p95 {stats['p95']:.3f}  max {stats['max']:.3f}"
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .training.train import train
    from .utils import read_json

    cfg = _load(args)
    run_dir = train(cfg, force_data=args.force)
    metrics = read_json(run_dir / "metrics.json")
    print(f"run {cfg.run_id} -> {run_dir}")
    print(f"  architecture {cfg.model.name}  params {metrics['n_parameters']:,}")
    print(f"  epochs       {metrics['epochs_run']}  ({metrics['train_wall_seconds']:.1f}s)")
    print(f"  relative L2  {metrics['relative_l2']:.4f}")
    print(f"  median rel.  {metrics['median_relative_error']:.4f}")
    print(f"  low-shear    dice {metrics['low_shear_dice']:.4f}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .training.evaluate import benchmark

    cfg = _load(args)
    runs = _latest_runs(cfg) if args.all_runs else _latest_runs(cfg)[-1:]
    report = benchmark(cfg, run_dirs=list(runs), split=args.split)
    print(report.read_text())
    print(f"written to {report}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .data import load_context, load_split
    from .data.generate import generate_dataset
    from .models.base import VesselContext
    from .models.baselines import PoiseuilleBaseline
    from .serving.bench import run_benchmark

    cfg = _load(args)
    dataset_dir = generate_dataset(cfg)
    features, _, _ = load_split(dataset_dir, "test")
    arrays = load_context(dataset_dir, "test")
    context = VesselContext(
        flow_rate=arrays["flow_rate"],
        inlet_radius=arrays["inlet_radius"],
        viscosity=cfg.fluid.viscosity_pa_s,
    )

    models = [PoiseuilleBaseline()]
    runs = _latest_runs(cfg)
    if runs:
        from .models.registry import load_surrogate

        surrogate, _ = load_surrogate(runs[-1])
        models.append(surrogate)
    else:
        print("no trained run found; benchmarking the analytic baseline only", file=sys.stderr)

    payload = run_benchmark(cfg, models, features, context)
    for row in payload["rows"]:
        print(
            f"{row['model']:<28} p50 {row['p50_ms']:7.2f} ms   "
            f"p95 {row['p95_ms']:7.2f} ms   batch {row['vessels_per_second']:8.1f} vessel/s"
        )
    print(
        f"\nSpeedup figures in reports/latency.json divide an ASSUMED "
        f"{payload['cfd_reference_seconds'] / 3600:.1f} h CFD solve by measured p50."
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    cfg = _load(args)
    print(json.dumps(to_plain(cfg), indent=2, sort_keys=True))
    print(f"\nrun_id     {cfg.run_id}")
    print(f"dataset_id {cfg.dataset_id}")
    print(f"run_dir    {cfg.run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hemoflow", description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str, handler) -> argparse.ArgumentParser:  # noqa: ANN001
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--config", type=str, default=None, help="path to a YAML config")
        sub.add_argument("--quiet", action="store_true", help="suppress progress logging")
        sub.set_defaults(handler=handler)
        return sub

    data = add("data", "generate the synthetic dataset", cmd_data)
    data.add_argument("--force", action="store_true", help="regenerate even if present")

    train = add("train", "train a surrogate", cmd_train)
    train.add_argument("--force", action="store_true", help="regenerate the dataset first")

    evaluate = add("eval", "benchmark against baselines", cmd_eval)
    evaluate.add_argument("--split", default="test", choices=["train", "val", "test"])
    evaluate.add_argument("--all-runs", action="store_true", help="include every trained run")

    add("bench", "measure inference latency", cmd_bench)
    add("info", "print the resolved config", cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
