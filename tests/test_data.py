"""Dataset integrity: splits, determinism, and normalisation hygiene.

These are the tests that catch the failure mode where everything runs, nothing
raises, and the reported error is a fiction.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemoflow.config import Config, DataConfig, GeometryConfig
from hemoflow.data import Standardizer, TargetTransform, generate_dataset, load_split, make_splits
from hemoflow.utils import read_json


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        name="test",
        geometry=GeometryConfig(n_arc=32, n_theta=16),
        data=DataConfig(n_geometries=24, seed=5, root=str(tmp_path / "data")),
    )


def test_splits_are_disjoint_by_vessel():
    """No vessel may appear in more than one split.

    This is the leakage test. Splitting surface *nodes* instead of whole vessels
    would put neighbouring points of the same artery in both train and test, and
    the resulting test error would measure memorisation rather than
    generalisation to a new patient.
    """
    splits = make_splits(100, val_fraction=0.15, test_fraction=0.15, seed=0)

    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert not train & val
    assert not train & test
    assert not val & test
    assert len(train | val | test) == 100
    assert all(len(s) > 0 for s in (train, val, test))


def test_splits_are_deterministic_given_a_seed():
    assert make_splits(50, 0.2, 0.2, seed=7) == make_splits(50, 0.2, 0.2, seed=7)
    assert make_splits(50, 0.2, 0.2, seed=7) != make_splits(50, 0.2, 0.2, seed=8)


def test_splits_reject_impossible_fractions():
    with pytest.raises(ValueError):
        make_splits(4, val_fraction=0.5, test_fraction=0.5, seed=0)


def test_generation_is_reproducible(cfg):
    first = generate_dataset(cfg)
    features_a = np.load(first / "features.npy")
    targets_a = np.load(first / "targets.npy")

    second = generate_dataset(cfg, force=True)
    np.testing.assert_array_equal(features_a, np.load(second / "features.npy"))
    np.testing.assert_array_equal(targets_a, np.load(second / "targets.npy"))


def test_dataset_id_ignores_training_hyperparameters(cfg):
    """Changing the learning rate must not invalidate a generated dataset."""
    from dataclasses import replace

    other = replace(cfg, train=replace(cfg.train, lr=cfg.train.lr * 10))
    assert other.dataset_id == cfg.dataset_id
    assert other.run_id != cfg.run_id


def test_manifest_records_provenance(cfg):
    manifest = read_json(generate_dataset(cfg) / "manifest.json")
    assert manifest["dataset_id"] == cfg.dataset_id
    assert manifest["split_sizes"]["train"] > 0
    assert "reduced-order" in manifest["reference_model"]
    assert len(manifest["feature_names"]) == manifest["shapes"]["features"][-1]


def test_targets_are_physically_plausible(cfg):
    _, targets, _ = load_split(generate_dataset(cfg), "train")
    assert (targets > 0).all(), "wall shear stress is a magnitude and cannot be negative"
    assert np.isfinite(targets).all()
    assert 0.05 < float(np.median(targets)) < 20.0, "median WSS outside any arterial range"


def test_standardizer_survives_a_constant_channel():
    """A zero-variance feature must not produce NaNs."""
    features = np.random.default_rng(0).normal(size=(4, 8, 8, 3))
    features[..., 1] = 2.5
    scaler = Standardizer.fit(features)
    out = scaler.transform(features)
    assert np.isfinite(out).all()
    assert np.allclose(out[..., 1], 0.0)


def test_standardizer_roundtrips_through_json(tmp_path):
    features = np.random.default_rng(1).normal(size=(3, 4, 4, 5))
    scaler = Standardizer.fit(features)
    path = scaler.save(tmp_path / "scaler.json")
    restored = Standardizer.load(path)
    np.testing.assert_allclose(scaler.transform(features), restored.transform(features))


def test_target_transform_is_invertible():
    values = np.array([0.05, 0.5, 2.0, 17.0, 60.0])
    transform = TargetTransform.fit(values, log_space=True)
    np.testing.assert_allclose(transform.inverse(transform.forward(values)), values, rtol=1e-5)


def test_log_space_equalises_relative_error_across_scales():
    """The reason for training in log space, asserted rather than assumed.

    A 10% error at 0.3 Pa and a 10% error at 30 Pa must cost the same, otherwise
    the loss silently abandons the low-shear regions that matter clinically.
    """
    transform = TargetTransform.fit(np.array([0.3, 30.0]), log_space=True)
    small = abs(transform.forward(np.array([0.33]))[0] - transform.forward(np.array([0.30]))[0])
    large = abs(transform.forward(np.array([33.0]))[0] - transform.forward(np.array([30.0]))[0])
    np.testing.assert_allclose(small, large, rtol=1e-5)
