# Evaluation: baselines, error distribution, failure cases

**Owner:** Bowen.

## Why relative L2 rather than MAE

Wall shear stress spans a factor of ~580 across this dataset (0.160 Pa to 93.5
Pa). A node-averaged metric like MAE is dominated by the handful of high-shear
throat nodes, so a model can post an excellent MAE while being useless in the
low-shear regions - which are the clinically interesting ones, because
persistently low and oscillating shear is what drives atherosclerotic plaque.
`relative_l2` and `median_relative_error` weigh a 10% error the same at 0.3 Pa
as at 30 Pa.

The MAE column in the benchmark makes the point by itself: `mlp` and `unet` are
0.0595 Pa apart in the fourth decimal place, while their `peak_median_relative_error`
differs by 18%. MAE cannot see that.

## Why low-shear Dice

`compute_metrics` thresholds both fields at `LOW_SHEAR_THRESHOLD_PA = 1.0` and
takes Dice overlap between the masks. It asks *where* the at-risk territory is -
the question a clinician actually has - rather than the pascal value at node
4,177. The constant baseline scores 0.4156 on it while posting a superficially
unremarkable MAE, which is exactly the failure the metric exists to expose.

## The full benchmark

`hemoflow eval --config configs/mlp.yaml --all-runs`, test split, dataset `9a3a9e28fc`:

| model | params | relative_l2 | median rel. err | low-shear Dice |
| --- | ---: | ---: | ---: | ---: |
| unet | 490,001 | 0.03063 | 1.83% | 0.9259 |
| mlp | 35,585 | 0.03096 | 1.88% | 0.9276 |
| poiseuille | 0 | 0.27 | 18.44% | 0.6797 |
| mlp (no physics prior) | 35,585 | 0.2867 | 18.37% | 0.6856 |
| ridge | 13 | 0.3267 | 21.35% | 0.6761 |
| constant | 0 | 0.7555 | 47.69% | 0.4156 |

Baselines are fitted on the train split only, inside `build_baselines`. Fitting
a "baseline" on test data is a quiet way to make a neural model look better than
it is.

Three things this table says:

1. **The U-Net earns nothing.** 490,001 parameters against 35,585 - 13.8x the
   model - for a 1.1% improvement in relative L2, and it is *behind* the MLP on
   low-shear Dice (0.9259 vs 0.9276) and on peak error (0.0381 vs 0.0322). The
   explanation is in the feature set: `throat_ratio` and `downstream_of_throat`
   already carry the non-local information a convolution would have to
   rediscover, so spatial context has little left to add.
2. **Both non-physics learned models lose to the analytic baseline.** Poiseuille
   at 0.27 beats the ablation (0.2867) and ridge (0.3267). Learning is only
   worth doing here on top of the physics, not instead of it.
3. **The headroom is nearly gone.** At `noise_sigma = 0.02` the label-noise
   floor is about 1.35% median relative error; the MLP is at 1.88%. Further
   accuracy against *this* reference is not where the remaining value is - a
   real CFD reference is.

## Where it does worst

`python scripts/worst_vessel.py`, MLP checkpoint, 77 test vessels:

    median rel-L2 0.0260   worst 0.0483

    rank  idx   rel-L2  stenosis   curv*r   peak Pa
       1   56   0.0483    53.3%   0.0750     17.25
       2   22   0.0423    42.7%   0.0536     24.58
       3   41   0.0414    55.7%   0.0485     12.56
       4   32   0.0412    41.1%   0.1219      7.87
       5   61   0.0404    58.6%   0.0449     21.78

Severity is not what drives the error. The correlation between stenosis
fraction and per-vessel relative L2 is only +0.248, and vessels at 52.9% and
46.1% narrowing sit in the best five - inside the range spanned by the worst
five (41-59%). Peak shear separates them far more cleanly: the worst five peak
at 12.6-24.6 Pa, the best five at 3.6-8.1 Pa. The error tracks the dynamic
range within a vessel rather than the depth of the throat, which is consistent
with where the reference solver works hardest - the separation and convective
corrections around a high-shear throat.

Vessel 32 is the exception worth naming: rank 4 by error, but the lowest peak
of that group at 7.87 Pa and roughly 2.4x the curvature of its neighbours. That
is a second failure mode, driven by the Dean-flow term rather than the throat.

The honest framing, whatever the detail turns out to be: the worst vessel is
only 1.86x the median, so there is **no catastrophic failure mode on this
dataset** - and that is a weaker result than it sounds. The vessels are
synthetic, they occupy a single calibre band, and the targets come from a
reduced-order reference rather than a 3D solve. Absence of a failure case here
is not evidence of robustness on real anatomy.

## Verified

- [x] Can explain why relative L2 and Dice rather than MAE
- [x] Ran `hemoflow eval --all-runs`, full table above
- [x] Ran `scripts/worst_vessel.py`, ranking above
- [x] Explained *why* the worst vessel is worst, having checked the comparison
