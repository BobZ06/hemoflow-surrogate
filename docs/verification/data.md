# Data: generation, leakage-safe splits, normalization

**Owner:** Bowen. Verifies `src/hemoflow/data/generate.py`, `data/dataset.py`, `data/normalize.py`.

## The generated dataset

`hemoflow data --config configs/mlp.yaml`:

    dataset 9a3a9e28fc -> artifacts/data/9a3a9e28fc
      vessels      {'test': 77, 'train': 358, 'val': 77}
      grid         [128, 64]
      WSS (Pa)     min 0.160  p50 1.946  p95 8.096  max 93.537

512 vessels on a 128 x 64 surface grid. Note the target range: a factor of ~580
between the quietest recirculation node and the tightest throat. That spread is
the reason for every metric choice in `training/metrics.py`.

## Why splits are by whole vessel

`make_splits` partitions vessel *indices*, never surface nodes. Two nodes a
millimetre apart on the same artery are nearly the same sample; if one landed in
train and its neighbour in test, the test score would measure interpolation
within a vessel the model has already half-seen, not generalisation to a new
patient. The number would look excellent and mean nothing.

## Why normalizers are fit on train only

`Standardizer.fit` and `TargetTransform.fit` run on the train split alone, then
apply unchanged to val and test. Fitting on the full dataset first would leak
test statistics into training. Both serialise to JSON and are written into the
run directory next to the weights by `save_checkpoint`, so the service loads the
exact scaling that was fitted rather than recomputing one - which is how
train/serve skew gets in.

`TargetTransform` works in log space on purpose: regressing raw pascals under L2
lets a handful of throat nodes dominate every gradient, and the model buys
accuracy there by giving up on the low-shear regions, which are the clinically
interesting ones.

## Why the targets carry noise

`data.noise_sigma = 0.02` applies multiplicative noise standing in for solver
discretisation error. Without it the target is a deterministic function of the
features and validation loss can be driven arbitrarily low, which would make the
model look better than any real deployment ever will. It also sets a floor:
median |exp(N(0, 0.02)) - 1| is about **1.35%**, so the MLP's 1.88% median
relative error sits at roughly 1.4x the label-noise floor. There is very little
headroom left on this reference.

## Reproducibility

`Config.dataset_id` hashes only `(geometry, fluid, data)`, so re-running
generation with an unchanged config is a no-op and changing a learning rate does
not invalidate the dataset.

**The command that proves it:** `pytest tests/test_data.py::test_generation_is_reproducible -v`.
It generates, regenerates with `force=True`, and asserts array equality on both
`features.npy` and `targets.npy`. That is a stronger claim than a one-off
checksum because it runs in CI on every push.

## Verified

- [x] Ran `hemoflow data --config configs/mlp.yaml` from scratch, read the manifest
- [x] Ran `pytest tests/test_data.py -v`: **11 passed**
- [x] Broke `make_splits` so the val slice overlaps the test slice: changed
      `order[n_test : n_test + n_val]` to `order[0 : n_val]`. The test failed on
      `assert not val & test` at `tests/test_data.py:39`, reporting the 15
      vessel ids present in both splits. Reverted, re-ran, green.
