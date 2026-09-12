# Architecture

## The shape of the problem

Given a vessel geometry and an inflow rate, predict wall shear stress at every
point on the lumen wall. The reference answer comes from computational fluid
dynamics, which costs hours per case. A surrogate learns the geometry-to-field
map offline and answers in milliseconds, which is what turns an overnight batch
job into something a clinician can move a slider on.

The pipeline is a straight line, and each stage depends only on the one before
it:

```
geometry  ->  data  ->  models  ->  training  ->  serving
```

## Why the representation is what it is

Handing a neural network the raw `(x, y, z)` coordinates of a vessel mesh forces
it to learn, from data, that a rotated artery is the same artery. With a few
hundred training geometries it will not learn that, and capacity spent on
orientation is capacity not spent on hemodynamics.

Instead every vessel is reduced to a centerline plus a radius field on an
`(arc x circumference)` grid, and features are expressed in a
**parallel-transport frame** carried along the centerline:

- A Frenet frame flips wherever curvature passes through zero, putting an
  artificial seam in the circumferential coordinate of a nearly straight vessel.
  Parallel transport advances by the minimal rotation between consecutive
  tangents, so it stays smooth.
- The frame's *starting angle* is still arbitrary - it is seeded from a world
  axis - so `build_vessel` anchors the circumferential grid to a
  curvature-weighted circular mean of the bend direction. That cancels the
  arbitrary offset exactly, and rotation invariance then holds element-wise on
  the discrete grid, not merely for the underlying continuous field.

`tests/test_geometry.py::test_features_are_rotation_invariant` asserts it
directly. Rotate a vessel, re-extract, compare node for node.

## Why the network predicts a ratio, not pascals

Every feature is dimensionless, which is what makes the representation
transferable between vessel calibres. But wall shear stress is not:

```
tau = 4 mu Q / (pi r^3) x (corrections)
```

and dimensionless features cannot recover absolute scale - the Reynolds number
carries `Q / r`, while shear needs `Q / r^3`. A network asked for pascals from
those features alone learns the average calibre of its training set.

This was a real bug here, not a hypothetical. A service test fed the model a
2.5-micron vessel and got back a perfectly ordinary 2 Pa.

So the network predicts the dimensionless ratio `tau_true / tau_poiseuille`, and
absolute scale comes from the analytic solution, which knows the real radius:

```
prediction_pa = network(features) x poiseuille(radius, flow)
```

Consequences: scale is exact at any calibre; the learning target is
well-conditioned near 1 instead of spanning two decades; and the Poiseuille
baseline is exactly the model that predicts a ratio of 1, so the benchmark table
reads as *what the network adds to textbook physics*.
`model.physics_residual: false` turns it off for the ablation.

## Why the convolutions pad the way they do

The unwrapped field is a cylinder, not a rectangle. The circumferential axis
wraps; the axial axis has real inlet and outlet boundaries. `MixedPadConv2d` uses
circular padding around the circumference and replicate padding along the axis.

Zero-padding both, which is what a stock `Conv2d` does, invents a discontinuity
along the `theta = 0` seam - and the Dean effect puts a smooth circumferential
wave in the target that the seam cuts straight through.

The decoder needed the same treatment. A plain `F.interpolate` clamps at the
tensor edge and, measured on a depth-2 net, reintroduced a ~30% discontinuity
across the seam that the padded convolutions had removed.
`topology_aware_upsample` pads with the correct topology before interpolating.

## The reference solver, and how to replace it

`geometry/reference.py` is **not** a 3D CFD solve. It is a reduced-order model:
a Poiseuille core with corrections for convective acceleration, post-stenotic
separation (a causal filter along the centerline, since separation is
history-dependent), entrance development, and Dean-type secondary flow.

It exists so the pipeline, metrics, service and demo could be built and validated
before any real solver output existed. Every metric in this repo measures
*fidelity to that reference*, not clinical accuracy, and the README says so.

Replacing it is a one-function change, because nothing downstream knows how the
targets were produced:

1. Write a loader that reads real solver output into `(n_arc, n_theta)` arrays on
   the same grid.
2. Swap the `reference_wss(...)` call in `data/generate.py::generate_dataset`.
3. Regenerate. The dataset hash changes, so a new directory appears and old runs
   keep pointing at the data that produced them.

Features, models, training, metrics and the service are untouched.

## Reproducibility

- One YAML config fully determines a run. No result-affecting flags.
- `Config.run_id` is a hash of the whole config; `Config.dataset_id` hashes only
  the geometry, fluid and data blocks, so changing a learning rate does not
  invalidate a generated dataset.
- Datasets and runs are content-addressed directories, written once. Re-running
  an unchanged config is a no-op rather than a silent second copy.
- Every manifest records the git SHA and a UTC timestamp.
- Splits are made over whole vessels, never surface nodes - two nodes a
  millimetre apart on the same artery are nearly the same sample, and splitting
  them across train and test would measure memorisation.
- Feature and target normalisers are fitted on train only and persisted next to
  the weights, so the service cannot re-derive them differently.

## Known limitations

- The reference is reduced-order. The headline numbers are fidelity to it.
- Synthetic geometries are single-branch tubes. No bifurcations, no aneurysm
  sacs, no side branches.
- Steady flow only. Real hemodynamics is pulsatile, and the clinically important
  oscillatory shear index needs a time-resolved solve.
- The API's vessel spec is a radius profile plus one constant curvature, not a
  full centerline.
