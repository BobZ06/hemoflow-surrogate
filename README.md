# hemoflow-surrogate

A neural surrogate for hemodynamic wall shear stress, built as an ML engineering
project. Vascular geometry is the application domain; the substance is
representation design, a physics-informed target, leakage-safe evaluation, and a
reproducible pipeline from data generation to a served endpoint.

```
geometry  ->  data  ->  models  ->  training  ->  serving
```

## The problem

Wall shear stress - the tangential drag blood exerts on an artery wall - is a
field you cannot measure directly and must simulate. Computational fluid
dynamics gives the answer and costs hours of CPU per case, which puts it firmly
in the overnight-batch category and out of reach of anything interactive.

A surrogate pays that cost once, offline, to build a training set, then
approximates the geometry-to-field map in milliseconds. This repo is a complete,
tested implementation of that idea with the evaluation machinery to say honestly
how well it works.

> **What the numbers below mean.** The reference fields come from a
> **reduced-order solver** (`geometry/reference.py`), not a 3D Navier-Stokes
> solve. It is a Poiseuille core with physically motivated corrections for
> convective acceleration, post-stenotic separation, entrance development and
> Dean-type secondary flow. Every metric here measures *fidelity to that
> reference*. None of it is a clinical accuracy claim. The solver sits behind a
> one-function interface precisely so it can be swapped for real CFD output
> without touching anything downstream - see
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quickstart

```bash
bash scripts/setup.sh     # CPU torch + this package
make smoke                # data -> train -> eval -> bench, ~40s
```

Then the real thing:

```bash
hemoflow data  --config configs/mlp.yaml     # generate the dataset
hemoflow train --config configs/mlp.yaml     # train and score on held-out test
hemoflow eval  --config configs/mlp.yaml --all-runs   # benchmark vs. baselines
hemoflow bench --config configs/mlp.yaml     # inference latency
make serve                                    # FastAPI on :8000, docs at /docs
```

<!-- RESULTS:START -->
Results are written to `artifacts/reports/` by `hemoflow eval` and
`hemoflow bench`.
<!-- RESULTS:END -->

## Three decisions that carry the project

Each of these started as a bug that produced plausible-looking wrong numbers.
Each is now pinned by a test.

**The features are rotation-invariant by construction, not by training.**
Handing a network raw `(x, y, z)` makes it learn from data that a rotated artery
is the same artery, and with a few hundred geometries it will not. Vessels are
reduced to a centerline plus a radius field on an `(arc x circumference)` grid,
with features expressed in a parallel-transport frame whose circumferential
origin is anchored to the vessel's own bend. Rotate a vessel and every feature
is unchanged node for node, which
`test_geometry.py::test_features_are_rotation_invariant` asserts directly.

**The network predicts a dimensionless ratio, not pascals.** Dimensionless
features cannot recover absolute scale: the Reynolds number carries `Q / r`
while shear needs `Q / r^3`. A network asked for pascals just learns its training
calibre - here it confidently returned an ordinary 2 Pa for a 2.5-micron vessel.
So the network predicts `tau_true / tau_poiseuille` and the analytic solution
supplies the scale. Absolute accuracy then holds at any calibre, the target is
well-conditioned near 1 instead of spanning two decades, and the Poiseuille
baseline becomes exactly "predict a ratio of 1" - so the benchmark reads as what
the network adds to textbook physics. `model.physics_residual: false` runs the
ablation.

**The convolutions know the field is a cylinder.** The circumferential axis
wraps; the axial axis has real inlet and outlet boundaries. Zero-padding both -
the stock `Conv2d` default - invents a seam at `theta = 0` that cuts straight
through the Dean-flow variation the model is supposed to learn. `MixedPadConv2d`
pads circularly in theta and by replication along the arc. The decoder needed the
same fix: a plain `F.interpolate` clamps at the edge and reintroduced a ~30%
discontinuity across that seam.

## Evaluation, and why these metrics

Mean absolute error over every node is a bad primary metric here. The target
spans two orders of magnitude, so node-averaged error is dominated by a handful
of stenosis-throat nodes, and a model can post an excellent MAE while being
useless in the low-shear regions - which are the ones that matter, since
persistently low shear is what drives plaque.

So the table carries three kinds of number: scale-free accuracy (relative L2,
median relative error), the same errors restricted to the sub-1 Pa atheroprone
band, and Dice agreement between the predicted and reference low-shear regions -
the closest thing here to the question actually being asked, which is *where* the
at-risk territory is.

Baselines are fitted on the training split only and scored through the same
`Surrogate.predict_pa` interface as the networks, so the comparison is
apples-to-apples.

## Layout

```
src/hemoflow/
  config.py          typed, hashable config; one YAML fully determines a run
  utils.py           seeding, fingerprinting, provenance
  geometry/          vessel representation, frames, features, reference solver
  data/              generation, leakage-safe splits, normalisation, loaders
  models/            interface, baselines, physics prior, networks, checkpoints
  training/          training loop, benchmark harness, metrics
  serving/           FastAPI service, latency benchmark
configs/             one file per experiment
tests/               57 tests; properties, not smoke
scripts/             setup, end-to-end smoke, figures
docs/                architecture, roadmap
```

## Reproducibility

- One YAML config fully determines a run; there are no result-affecting flags.
- Datasets and runs are content-addressed by config hash and written once, so
  re-running an unchanged config is a no-op rather than a silent second copy.
  `dataset_id` hashes only the geometry, fluid and data blocks, so changing a
  learning rate does not invalidate a generated dataset.
- Splits are over whole vessels, never surface nodes. Two nodes a millimetre
  apart on the same artery are nearly the same sample, and splitting them across
  train and test would measure memorisation.
  (`test_data.py::test_splits_are_disjoint_by_vessel`)
- Normalisers are fitted on train only and persisted next to the weights, so a
  run directory is servable on its own and the service cannot re-derive them
  differently. (`test_models.py::test_checkpoint_roundtrip_preserves_predictions`)
- Every manifest records the git SHA and a UTC timestamp.

## Limitations

The reference is reduced-order, not CFD. Geometries are synthetic single-branch
tubes - no bifurcations, no aneurysm sacs, no side branches. Flow is steady,
while real hemodynamics is pulsatile and the clinically interesting oscillatory
shear index needs a time-resolved solve. The model reports no uncertainty, which
is the gap that matters most before anything like this goes near a decision.

[`docs/ROADMAP.md`](docs/ROADMAP.md) has the ordered plan, including what was
deliberately left out and why.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup, the two-lane split of
ownership, branch conventions, and the failure modes specific to this domain
(units, leakage, stale caches).

```bash
make fmt lint test
```
