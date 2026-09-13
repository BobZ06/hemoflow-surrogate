# Checkpoint loading and round-trip verification

**Primary owner:** Letian

This verification covers `src/hemoflow/models/registry.py`, especially
`save_checkpoint` and `load_surrogate`.

## Why the run directory is self-contained

A checkpoint needs more than the neural-network weights. The complete run
directory contains:

- `weights.pt`: trained model parameters
- `standardizer.json`: input-feature normalization state
- `target.json`: target transformation state
- `config.json`: architecture and experiment configuration
- `metrics.json`: evaluation metrics
- `meta.json`: run and dataset metadata
- `history.json`: training and validation history

The standardizer must be saved with the weights. If it were missing, the
reloaded model could receive inputs on a different numerical scale and produce
different predictions.

The target transform must also be saved because the network is trained on a
transformed target. During prediction, the saved transform is reversed before
returning the final WSS values.

The configuration is required to rebuild the same network architecture before
loading `weights.pt`.

## Checkpoint contents

The verified checkpoint is `artifacts/runs/mlp-53e1ed5fd3`.

It contains:

`config.json`, `history.json`, `meta.json`, `metrics.json`,
`standardizer.json`, `target.json`, and `weights.pt`.

## Round-trip test

The save-and-reload prediction test was run with:

`.\.venv\Scripts\python.exe -m pytest tests/test_models.py::test_checkpoint_roundtrip_preserves_predictions -v`

Result: `1 passed in 72.36s`

This test confirms that predictions remain consistent after the model is saved
and reloaded.

## Fresh-process loading

The checkpoint was loaded from a new Python process with:

`.\.venv\Scripts\python.exe -c "from hemoflow.models.registry import load_surrogate; s,c=load_surrogate('artifacts/runs/mlp-53e1ed5fd3'); print(s.name, s.n_parameters, c.run_id)"`

Result: `mlp 35585 mlp-53e1ed5fd3`

This confirms that the checkpoint can be reconstructed from its saved files
without the original Python process.

## Verification status

- [x] Checkpoint contains weights, normalization state, target transform, and configuration.
- [x] Round-trip prediction test passed.
- [x] Checkpoint loaded successfully in a fresh Python process.
- [ ] Loading on another developer's machine remains a separate cross-machine check.