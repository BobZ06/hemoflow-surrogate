# Physics prior: the residual formulation and its ablation

**Owner:** Bowen (explanation and verification). Letian confirms the model-side
wiring and checkpoint loading respect `cfg.model.physics_residual`.

## What the network actually predicts

Not pascals. The network predicts the dimensionless ratio

    ratio = tau_true / tau_poiseuille

and `poiseuille_field(...)` - computed from the radius and flow, which are known
exactly - supplies the absolute scale afterwards. Three consequences, all good:
scale is exact by construction at any calibre; the learning target is a
well-conditioned quantity near 1 instead of one spanning two orders of
magnitude; and the Poiseuille baseline becomes precisely the model that predicts
`ratio = 1`, so the benchmark table reads as "what the network adds to textbook
physics".

## The result, from `hemoflow eval --config configs/mlp.yaml --all-runs`

| model | params | rel. L2 | median rel. err | low-shear Dice |
| --- | ---: | ---: | ---: | ---: |
| mlp (physics prior) | 35,585 | 0.03096 | 1.88% | 0.9276 |
| poiseuille (analytic, no learning) | 0 | 0.27 | 18.44% | 0.6797 |
| mlp (no physics prior) | 35,585 | 0.2867 | 18.37% | 0.6856 |

Two numbers to be able to state cold:

**9.3x.** Identical architecture, identical parameter count, identical data -
turning the prior off costs a factor of 9.3 in relative L2 (0.2867 / 0.03096).
The entire gap is what happens to the `1/r^3` factor.

**The ablation loses to the analytic baseline.** 0.2867 against Poiseuille's
0.27. A 35,585-parameter network, fully trained, does *worse than the textbook
formula with zero parameters*. That is the sharper claim, and it is the honest
one: asked to emit absolute pascals from dimensionless inputs, the network
spends its capacity on a scale it cannot represent and ends up behind doing
nothing. With the prior, the same network beats that baseline by 8.7x.

## Verified live in the demo, not just in a training log

Toggling the demo's physics-prior switch changes which environment variable
`serving/demo.py` reads - `HEMOFLOW_RUN_DIR` vs `HEMOFLOW_ABLATION_RUN_DIR` -
which loads a different checkpoint entirely, not the same model relabelled.
`tests/test_demo.py::test_physics_prior_keeps_absolute_scale_four_decades_out`
and `::test_ablation_loses_that_scale` pin both halves: halving inlet diameter
must raise peak shear ~8x with the prior on, and must visibly fail to track that
scaling with it off.

## Verified

- [x] Trained `configs/mlp.yaml` and `configs/mlp_no_physics.yaml`
- [x] Ran `hemoflow eval --all-runs`, table above
- [x] Can state 9.3x, and the stronger Poiseuille comparison, from memory
- [x] Confirmed in the running demo that the toggle loads a different checkpoint,
      not a relabelled output: identical request to /demo/predict at 0.4 mm inlet
      returns 36.78 Pa with the prior on and 0.76 Pa with it off (48x), and the
      low-shear fraction flips from 0.0 to 1.0. `model_name` is "mlp" in both
      responses, so the label alone proves nothing - the behavioural split is the
      evidence. /demo/status confirms run_id mlp-53e1ed5fd3, 35,585 parameters.
