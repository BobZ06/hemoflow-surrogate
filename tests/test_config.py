"""Config loading, hashing, and the round-trip that has to be lossless."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import yaml

from hemoflow.config import Config, load_config
from hemoflow.utils import to_plain, write_json


def test_defaults_validate():
    Config().validate()


def test_run_id_is_stable_and_config_dependent():
    a, b = Config(), Config()
    assert a.run_id == b.run_id
    assert replace(a, seed=a.seed + 1).run_id != a.run_id


def test_dataset_id_ignores_model_and_training_blocks():
    cfg = Config()
    assert replace(cfg, train=replace(cfg.train, lr=1.0)).dataset_id == cfg.dataset_id
    assert replace(cfg, model=replace(cfg.model, depth=9)).dataset_id == cfg.dataset_id
    assert replace(cfg, data=replace(cfg.data, seed=99)).dataset_id != cfg.dataset_id


def test_json_roundtrip_is_lossless(tmp_path):
    """Write a config, read it back, and get the same hash.

    The bug this pins: `json.dumps` writes 8e-6 as `8e-06`, and YAML 1.1's float
    resolver requires a decimal point in the mantissa, so the YAML loader
    returned the *string* `"8e-06"` while `1.5e-06` came back as a float. The
    config silently changed identity on a round-trip, which made every run look
    like it had been trained on a different dataset than it was.
    """
    cfg = Config(name="roundtrip")
    path = write_json(tmp_path / "config.json", cfg)
    restored = load_config(path)

    assert restored == cfg
    assert restored.run_id == cfg.run_id
    assert restored.dataset_id == cfg.dataset_id


@pytest.mark.parametrize("literal", ["8e-06", "8.0e-6", "0.000008", "8E-06"])
def test_exponent_notation_always_parses_as_float(tmp_path, literal):
    """Every spelling of a small float must load as a float, not a string."""
    path = tmp_path / "c.yaml"
    path.write_text(f"fluid:\n  flow_max_m3_s: {literal}\n")
    value = load_config(path).fluid.flow_max_m3_s
    assert isinstance(value, float)
    assert value == pytest.approx(8e-6)


def test_yaml_roundtrip_is_lossless(tmp_path):
    cfg = Config(name="yaml-roundtrip")
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(to_plain(cfg)))
    assert load_config(path) == cfg


def test_scalars_are_cast_to_their_declared_types(tmp_path):
    """A config written by hand with loose types still loads correctly typed."""
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            {
                "geometry": {"n_arc": "64", "n_theta": 32},
                "train": {"epochs": "5", "lr": "1e-3", "log_space": "false"},
                "model": {"hidden_dims": ["16", "32"]},
            }
        )
    )
    cfg = load_config(path)

    assert cfg.geometry.n_arc == 64 and isinstance(cfg.geometry.n_arc, int)
    assert cfg.train.lr == pytest.approx(1e-3) and isinstance(cfg.train.lr, float)
    assert cfg.train.log_space is False
    assert cfg.model.hidden_dims == (16, 32)
    assert all(isinstance(d, int) for d in cfg.model.hidden_dims)


def test_lists_become_tuples_so_the_config_stays_hashable(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  hidden_dims: [8, 8]\n")
    assert isinstance(load_config(path).model.hidden_dims, tuple)


def test_unknown_keys_are_rejected(tmp_path):
    """A typo in a config must fail loudly, not be silently ignored."""
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  learning_rate: 0.01\n")
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)


def test_invalid_configs_are_rejected_before_anything_runs(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("data:\n  val_fraction: 0.6\n  test_fraction: 0.5\n")
    with pytest.raises(ValueError, match="leave room for training data"):
        load_config(path)

    path.write_text("model:\n  name: transformer\n")
    with pytest.raises(ValueError, match="unknown model"):
        load_config(path)


def test_shipped_configs_all_load():
    from pathlib import Path

    configs = sorted(Path("configs").glob("*.yaml"))
    assert configs, "no configs found - run from the repository root"
    for path in configs:
        load_config(path).validate()
