# Training: experiment configuration, runs, and records

**Owner:** Bowen.

## Early stopping and checkpoint selection

`train()` tracks validation loss every epoch and keeps a detached CPU copy of
the state dict whenever it improves by more than `1e-6`. After
`cfg.train.patience` epochs without improvement it stops early. The *best*
state - never the last epoch's weights - is reloaded before test scoring, so a
model that starts overfitting late is not graded at its worst point.

The copy is detached and moved to CPU deliberately: holding GPU tensors alive
across sixty epochs is how a long sweep runs out of memory at epoch 40.

## `Config.run_id` vs `Config.dataset_id`

- `run_id = f"{name}-{fingerprint(self)}"` hashes the **whole** config. Change
  the learning rate and you get a different run directory, because that is
  genuinely a different experiment.
- `dataset_id = fingerprint((geometry, fluid, data))` hashes **only** the blocks
  that determine the generated data. Changing `train.lr` must not invalidate an
  already-generated dataset; regenerating 512 vessels because a learning rate
  moved would be absurd.

`evaluate.py::load_runs` leans on this: a run whose `dataset_id` differs from
the one being evaluated is skipped and named in the report rather than silently
included, because the same column headings over different held-out vessels is a
table that looks fine and means nothing.

## The experiment log

All three trained on Apple Silicon (`device=mps`), same dataset `9a3a9e28fc`:

| config | run_id | params | epochs | wall | rel. L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| `configs/mlp.yaml` | `mlp-53e1ed5fd3` | 35,585 | 44 | 44.7s | 0.0310 |
| `configs/mlp_no_physics.yaml` | `mlp-no-physics-d30ab0a45e` | 35,585 | 57 | 54.2s | 0.2867 |
| `configs/unet.yaml` | `unet-8e11d62f09` | 490,001 | 18 | 28.1s | 0.0306 |

Reproduce any row with `hemoflow train --config <that config>`. The run
directory name is derived from the config, so the same config on a different
machine writes the same `run_id` - which is the reproducibility claim, and it is
checkable in one command rather than asserted.

The U-Net ran its full 18-epoch budget without early stopping (val loss still
improving at epoch 18, 0.01248). The MLP stopped at 44 of 60; the ablation at 57
of 60 - it was still grinding at a problem it cannot solve.

## Verified

- [x] Can explain early stopping and best-checkpoint selection
- [x] Can explain `run_id` vs `dataset_id` and why they hash different fields
- [x] Ran the full training matrix; run directories above
