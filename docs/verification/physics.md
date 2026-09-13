# Physics: WSS formula, units, and scale relationships

**Owner:** Bowen. Verifies `src/hemoflow/models/physics.py`.

## The formula

    tau = 4 * mu * Q / (pi * r^3)

- `mu`: blood viscosity, `FluidConfig.viscosity_pa_s = 0.0035 Pa*s`
- `Q`: volumetric flow rate, m^3/s
- `r`: local lumen radius, m

## Worked example

r = 2 mm, Q = 4 mL/s (mid-range of `FluidConfig.flow_min/max_m3_s`):

    tau = 4 * 0.0035 * 4e-6 / (pi * 0.002^3) = 5.6e-8 / 2.513e-8 = 2.23 Pa

Halve the radius to 1 mm at the same flow:

    tau = 5.6e-8 / (pi * 0.001^3) = 17.8 Pa

An 8x jump for a 2x radius drop, because r enters cubed. That factor of 8 is
what `tests/test_demo.py::test_physics_prior_carries_absolute_scale_across_calibres`
asserts against the live endpoint.

Sanity check against the generated dataset: `hemoflow data --config
configs/mlp.yaml` reports a p50 of 1.946 Pa over 512 vessels, which is the
right order for the 2-4 mm calibre band the geometry sampler draws from.

## Why dimensionless features cannot recover this

Every entry in `FEATURE_NAMES` is a ratio - `log_radius_ratio` is `r / inlet_r`,
never `r` itself. Reynolds number carries `Q / r` (`velocity = Q/(pi r^2)`,
`Re = rho * v * D / mu`). Shear needs `Q / r^3`. No combination of the twelve
dimensionless features can reconstruct the missing two powers of r, because
absolute scale is exactly what a dimensionless quantity discards.

## The 2.5-micron-vessel bug

The no-physics-prior model (`configs/mlp_no_physics.yaml`) emits pascals
directly from the dimensionless features. Fed a request scaled to a 2.5-micron
capillary - four decades below the training band - it returns an ordinary
few-pascal answer, because it learned the average calibre of its training set
rather than the physics. `poiseuille_field(...)` on the same input returns
numbers in the thousands of pascals, consistent with `tau ~ 1/r^3`.

The fix is to let the network predict only the dimensionless ratio
`tau_true / tau_poiseuille` and multiply the analytic field back in at the end,
so absolute scale is exact by construction at any calibre.

## Verified

- [x] Derived the formula and both worked numbers above
- [x] Ran `pytest -k "poiseuille or physics" -v`: **5 passed, 75 deselected**
      (1 in `test_api.py`, 1 in `test_demo.py`, 3 in `test_models.py`)
- [x] Hand-verified the Poiseuille scale factor against `poiseuille_field` for a
      vessel 10x smaller than training - the 1 mm case above

Note: issue #1 as filed says to run `pytest tests/test_models.py -k physics`.
That selects zero tests; no test in that file has "physics" in its name. The
command above is the correct one.
