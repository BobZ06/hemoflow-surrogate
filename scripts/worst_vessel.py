"""Find where the surrogate does worst on the held-out split, and characterise it.

Answers the failure-case item in the evaluation checklist. Per-vessel relative
L2 is ranked, and each vessel is described by two quantities recovered from its
own (dimensionless) features: how far the lumen narrows, and how hard the
centerline bends. Printing the best five alongside the worst five is the point -
a ranking with nothing to compare it against invites a causal story the data
does not support.
"""

import numpy as np

from hemoflow.config import load_config
from hemoflow.geometry.features import FEATURE_NAMES
from hemoflow.models.registry import load_surrogate
from hemoflow.training.metrics import relative_l2
from hemoflow.training.train import load_eval_batch

RUN_DIR = "artifacts/runs/mlp-53e1ed5fd3"

cfg = load_config("configs/mlp.yaml")
surrogate, _ = load_surrogate(RUN_DIR)
features, targets, context = load_eval_batch(cfg, split="test")
pred = surrogate.predict_pa(features, context)

per_vessel = np.array([relative_l2(pred[i], targets[i]) for i in range(len(targets))])
order = np.argsort(per_vessel)[::-1]

lr = FEATURE_NAMES.index("log_radius_ratio")
kr = FEATURE_NAMES.index("curvature_x_radius")


def describe(rank: int, i: int) -> None:
    stenosis = 1.0 - np.exp(features[i, :, :, lr].min())
    curv = features[i, :, :, kr].max()
    print(
        f"{rank:>4} {i:>4} {per_vessel[i]:>8.4f} {stenosis:>8.1%} "
        f"{curv:>8.4f} {targets[i].max():>9.2f}"
    )


print(f"test vessels: {len(per_vessel)}")
print(f"median rel-L2 {np.median(per_vessel):.4f}   worst {per_vessel.max():.4f}\n")

header = f"{'rank':>4} {'idx':>4} {'rel-L2':>8} {'stenosis':>9} {'curv*r':>8} {'peak Pa':>9}"

print("WORST FIVE")
print(header)
for rank, i in enumerate(order[:5], 1):
    describe(rank, i)

print("\nBEST FIVE")
print(header)
for rank, i in enumerate(order[-5:][::-1], len(order) - 4):
    describe(rank, i)

stenoses = np.array([1.0 - np.exp(features[i, :, :, lr].min()) for i in range(len(targets))])
corr = float(np.corrcoef(stenoses, per_vessel)[0, 1])
print(f"\ncorrelation between stenosis severity and rel-L2: {corr:+.3f}")
print(f"spread: worst / median = {per_vessel.max() / np.median(per_vessel):.2f}x")
