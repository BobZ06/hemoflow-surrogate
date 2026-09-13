# The demo

```bash
make train        # once; writes artifacts/runs/<run_id>/
make ablation     # once; the checkpoint behind the physics-prior switch
make demo         # http://127.0.0.1:8000/
```

It runs entirely on localhost with no network access of any kind. The page
loads no fonts, no CDN scripts and no remote assets, which
`test_demo.py::test_page_is_served_and_carries_no_external_requests` asserts by
scanning the served HTML - a demo that needs wifi is a demo that fails in the
one room where it matters.

## What is on screen

**Longitudinal section.** The vessel as a clinician would section it. The two
wall lines are the outer and inner wall of the bend, stroked in the field's own
colour, because wall shear stress is a quantity that lives on the wall. The
vermilion rule marks wall below 1 Pa - the atheroprone band, where low shear is
associated with plaque development. It is the only saturated colour on the page
that is not data, so red always means the same thing.

**Blood-flow animation.** Shaded red blood cells and faint streamlines occupy the lumen while
wall colours continue to represent WSS. Cells are enlarged illustrative biconcave
discs with subtle tumbling, rendered from cached Canvas sprites. Their size and
count do not represent a measured cell size or concentration. Particle transport uses cumulative
cross-sectional volume and the request flow rate, with a parabolic radial speed
profile: particles move faster through narrowing and more slowly near the wall.
The inlet and throat readouts show section-mean speed from `Q / (pi r^2)` in m/s,
and their ratio. This is an illustrative steady-flow approximation, not a
velocity field predicted by the surrogate or a CFD particle simulation; it does
not reproduce recirculation or pulsatility.

Pause/play and a 0.25–2× playback slider affect only the animation. At default
playback, physical time is slowed to 0.08×, with an additional bound for extreme
inputs; the actual factor is shown below the view. Reduced-motion preferences
start the animation paused, and hidden tabs stop requesting frames. Animation
uses its own canvas and never triggers model requests. Resizing redraws the
last successful result locally.

**Unwrapped lumen surface.** The `(arc x circumference)` grid the model actually
predicts on, for the surrogate and for the reference solver, on one shared
colour scale so a difference between the panels is a difference in the field and
not in how each was normalised. The third panel is the signed relative
relative error, `(surrogate - reference) / reference`, with a separate blue–white–orange
percentage legend (blue underestimates, orange overestimates). The default fixed
range is +/-10%, selectable from +/-2% to +/-100%. Values beyond the selected
range saturate; their fraction and median/max absolute relative errors are shown.
The error map uses unsmoothed grid cells and supports pointing/tapping for local
WSS values and signed error. The Pa legend applies only to the two WSS maps.

**Readouts.** Query time is feature extraction plus the forward pass, which is
what a caller actually pays; the speedup is quoted against the same assumed
4-hour CFD solve the benchmark harness uses, imported from
`serving/bench.py` rather than restated here so the two can never drift.

## The two moments worth showing

**The physics-prior switch.** Off, the same network with the same features and
the same training budget has to emit pascals directly from dimensionless inputs.
The banner reports relative L2 against the reference, which moves by roughly an
order of magnitude. This is the ablation from the README, made watchable.

**Capillary calibre.** The preset takes the vessel to 8 microns - four decades
below anything in training. With the prior on, absolute scale stays exact,
because it comes from the analytic solution rather than from the network. With
it off, the network returns something close to its training calibre no matter
what it is shown. `test_demo.py::test_physics_prior_carries_absolute_scale_across_calibres`
pins the `1/r^3` mechanism over one octave, 4 mm against 2 mm; the four-decade
behaviour is what the preset demonstrates, not something a test asserts.

## What the demo does not claim

The reference the panels compare against is `geometry/reference.py`, a
reduced-order solver, not a 3D Navier-Stokes solve. Every number on the page is
fidelity to that reference. The response field is named
`reduced_order_reference` rather than anything involving the letters CFD for
exactly that reason, and `docs/ARCHITECTURE.md` describes the one-function swap
that replaces it.

The vessel view projects circular cross-sections into elliptical open ends.
Rear and translucent front surfaces use cylindrical lighting, with a curved
specular highlight following the narrowing. Cells render between these surfaces;
WSS boundaries and low-shear markers render last to preserve their visibility.
Surface layers are cached when geometry changes rather than rebuilt each frame.
This is a fixed-view illustrative projection, not a patient-specific tissue mesh
or a new velocity simulation. The vessel remains straightened along its centerline.
