"""Dataset generation: vessels -> features and targets on disk, with a manifest.

A generated dataset is a build artifact, not source. It is addressed by a hash
of the config fields that actually determine its contents, written once, and
never edited in place. Re-running generation with an unchanged config is a
no-op; changing any geometry, fluid or data field produces a new directory
rather than silently overwriting the old one, so an old run's metrics always
still point at the data that produced them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Config
from ..geometry import FEATURE_NAMES, extract_features, reference_wss, sample_vessel
from ..utils import git_sha, set_seed, utc_now, write_json

SPLIT_NAMES = ("train", "val", "test")


def make_splits(
    n_items: int, val_fraction: float, test_fraction: float, seed: int
) -> dict[str, list[int]]:
    """Partition vessel indices into train/val/test.

    Splitting happens at the level of whole *vessels*, never surface nodes. Two
    nodes a millimetre apart on the same artery are almost the same sample; if
    one lands in train and the other in test, the test score measures
    interpolation within a geometry the model has already seen and the reported
    error is meaningless. `tests/test_data.py::test_splits_are_disjoint_by_vessel`
    pins this down.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    n_test = max(1, int(round(test_fraction * n_items)))
    n_val = max(1, int(round(val_fraction * n_items)))
    if n_test + n_val >= n_items:
        raise ValueError(f"{n_items} vessels cannot fill three non-empty splits")
    return {
        "test": sorted(int(i) for i in order[:n_test]),
        "val": sorted(int(i) for i in order[n_test : n_test + n_val]),
        "train": sorted(int(i) for i in order[n_test + n_val :]),
    }


def generate_dataset(cfg: Config, force: bool = False) -> Path:
    """Materialise the dataset described by `cfg` and return its directory.

    Args:
        cfg: full experiment config; only the geometry/fluid/data blocks affect
            the output, which is why `cfg.dataset_id` hashes those alone.
        force: regenerate even if a complete dataset is already present.

    Returns:
        Path to the dataset directory containing `features.npy`, `targets.npy`,
        `splits.json` and `manifest.json`.
    """
    out_dir = cfg.dataset_dir
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return out_dir

    set_seed(cfg.data.seed)
    rng = np.random.default_rng(cfg.data.seed)

    n = cfg.data.n_geometries
    features = np.zeros(
        (n, cfg.geometry.n_arc, cfg.geometry.n_theta, len(FEATURE_NAMES)), dtype=np.float32
    )
    targets = np.zeros((n, cfg.geometry.n_arc, cfg.geometry.n_theta), dtype=np.float32)
    vessel_ids: list[str] = []
    # Physical scale is deliberately absent from the (dimensionless) features, so
    # the two quantities a physics baseline needs are carried alongside them.
    flow_rates = np.zeros(n, dtype=np.float32)
    inlet_radii = np.zeros(n, dtype=np.float32)

    for i in range(n):
        vessel_id = f"synth-{cfg.data.seed}-{i:05d}"
        vessel = sample_vessel(rng, cfg.geometry, cfg.fluid, vessel_id)
        field = reference_wss(vessel, cfg.fluid)

        if cfg.data.noise_sigma > 0:
            # Multiplicative noise stands in for solver discretisation error.
            # Without it the target is a deterministic function of the features
            # and validation loss can be driven arbitrarily low, which would
            # make the model look better than any real deployment ever will.
            field = field * np.exp(rng.normal(0.0, cfg.data.noise_sigma, size=field.shape))

        features[i] = extract_features(vessel, cfg.fluid)
        targets[i] = field.astype(np.float32)
        flow_rates[i] = vessel.flow_rate
        inlet_radii[i] = vessel.radius[0]
        vessel_ids.append(vessel_id)

    splits = make_splits(n, cfg.data.val_fraction, cfg.data.test_fraction, cfg.data.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", features)
    np.save(out_dir / "targets.npy", targets)
    np.save(out_dir / "flow_rates.npy", flow_rates)
    np.save(out_dir / "inlet_radii.npy", inlet_radii)
    write_json(out_dir / "vessel_ids.json", vessel_ids)
    write_json(out_dir / "splits.json", splits)
    write_json(
        manifest_path,
        {
            "dataset_id": cfg.dataset_id,
            "created_utc": utc_now(),
            "git_sha": git_sha(),
            "geometry": cfg.geometry,
            "fluid": cfg.fluid,
            "data": cfg.data,
            "feature_names": list(FEATURE_NAMES),
            "shapes": {"features": list(features.shape), "targets": list(targets.shape)},
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "target_stats_pa": {
                "min": float(targets.min()),
                "p50": float(np.percentile(targets, 50)),
                "p95": float(np.percentile(targets, 95)),
                "max": float(targets.max()),
            },
            "reference_model": "reduced-order (see geometry/reference.py) - NOT a 3D CFD solve",
        },
    )
    return out_dir


def load_split(dataset_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load one split's features and targets.

    Returns:
        `(features, targets, indices)` where indices are positions in the full
        dataset, kept so predictions can be traced back to a specific vessel.
    """
    from ..utils import read_json

    dataset_dir = Path(dataset_dir)
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    splits = read_json(dataset_dir / "splits.json")
    idx = splits[split]
    features = np.load(dataset_dir / "features.npy", mmap_mode="r")
    targets = np.load(dataset_dir / "targets.npy", mmap_mode="r")
    return np.asarray(features[idx]), np.asarray(targets[idx]), idx


def load_context(dataset_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    """Load the physical scale of each vessel in a split.

    Features are dimensionless by design, which is what makes them transferable
    across vessel calibres - but a physics baseline needs real millimetres and
    real flow, so those live here rather than in the feature tensor.
    """
    from ..utils import read_json

    dataset_dir = Path(dataset_dir)
    idx = read_json(dataset_dir / "splits.json")[split]
    return {
        "flow_rate": np.load(dataset_dir / "flow_rates.npy")[idx],
        "inlet_radius": np.load(dataset_dir / "inlet_radii.npy")[idx],
    }
