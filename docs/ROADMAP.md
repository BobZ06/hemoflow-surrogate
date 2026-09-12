# Roadmap

What is built, what is deliberately not, and what to reach for next. Ordered by
value per hour, which during a build weekend is the only ordering that matters.

## Built

- Rotation-invariant vessel representation with parallel-transport frames, with
  the invariance asserted element-wise in tests.
- Reduced-order reference solver standing in for 3D CFD, behind a swappable
  interface.
- Config-addressed dataset generation with manifests, leakage-safe splits and
  reproducible seeds.
- Four models behind one interface: constant, analytic Poiseuille, ridge, and two
  networks (per-node MLP, topology-aware U-Net).
- Physics-residual formulation so absolute scale comes from the analytic solution
  and the network only learns the dimensionless correction.
- Domain-aware metrics: relative L2, plus low-shear-band accuracy and Dice
  agreement on the atheroprone region.
- Training loop with early stopping, best-checkpoint selection, and self-contained
  run directories.
- FastAPI inference service with validated clinical units and per-request timing.
- Latency benchmark separating interactive p50 from batched throughput.
- 57 tests, CI with lint, matrix tests and an end-to-end smoke run.

## Next, in order

**1. Real solver targets.** The single highest-value change. Everything else is
scaffolding for this. Swap `reference_wss` for a loader over real CFD output and
regenerate; see `docs/ARCHITECTURE.md`. Public sources worth trying first: the
Vascular Model Repository, which ships geometries with simulation results.

**2. Real geometries.** Replace synthetic tubes with segmented patient anatomy.
The path is CT/MR -> segmentation (TotalSegmentator is the fast option) -> surface
mesh -> centerline extraction (VMTK, or a distance-transform skeleton as a
fallback) -> resample onto the `(arc x circumference)` grid. Only the last step
touches this repo; the representation is already the interface.

**3. Bifurcations.** The current representation assumes a single tube, and the
clinically interesting lesions are disproportionately at branch points, where
flow divides and low-shear regions form on the outer walls. This is a real
representational change - an unwrapped grid does not survive a branch - and is
where a graph network earns its complexity over a U-Net.

**4. Pulsatile flow.** Steady flow gives one field. Real hemodynamics is
time-resolved, and the oscillatory shear index - how much shear direction reverses
over a cardiac cycle - is arguably a better plaque predictor than mean magnitude.
Requires time-series targets and a temporal model.

**5. Calibrated uncertainty.** A surrogate that cannot say when it is out of
distribution should not be near a clinical decision. A deep ensemble is the
cheapest credible option; quantile regression on the log-ratio is cheaper still.
This matters more than another point of accuracy.

## Deliberately not done

- **No GNN.** It would be the right choice for bifurcations and unstructured
  meshes, and the wrong one here: on a single tube the unwrapped grid is exactly
  the structure a CNN exploits, and a GNN would cost more code and more compute to
  represent the same neighbourhood.
- **No SE(3)-equivariant architecture.** Invariance is built into the features
  instead, which costs zero parameters and is testable in one assertion.
- **No mesh viewer.** The unwrapped field plus the error map answers "where is the
  model wrong" better than a 3D render does, at a fraction of the build cost.
- **No Docker image.** `pip install -e .` and a Makefile are enough for two people
  on two laptops. Containerising is the right call the moment it goes anywhere
  else.
