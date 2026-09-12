"""The physics prior: what the network is allowed to not learn.

Motivation, which is worth stating plainly because it caught a real bug in this
repo:

Every feature in `geometry/features.py` is dimensionless. That is what makes the
representation transferable across vessel calibres - a coronary and an iliac
with the same relative narrowing look the same to the model. But wall shear
stress is *not* dimensionless. It is

    tau = 4 mu Q / (pi r^3) * (corrections)

and the dimensionless features cannot pin down absolute scale: the Reynolds
number carries `Q / r`, while shear needs `Q / r^3`. A network asked to emit
pascals from those features alone has no way to recover the missing factor, so
it learns the average calibre of its training set and silently mispredicts by
orders of magnitude on anything else. The service test that feeds a 2.5-micron
vessel is what exposed this: it came back with a perfectly ordinary 2 Pa.

The fix is to give scale back to physics and leave only the hard part to the
network. The analytic Poiseuille field is computed from the radius and flow that
are known exactly, and the network predicts the dimensionless ratio

    ratio = tau_true / tau_poiseuille

which *is* a function of dimensionless geometry and Reynolds number. The two are
multiplied at the end.

Three things follow, all of them good:

* Absolute scale is exact by construction, at any calibre, in any unit system.
* The learning target is a well-conditioned quantity near 1 instead of a
  quantity spanning two orders of magnitude.
* The Poiseuille baseline becomes precisely the model that predicts `ratio = 1`,
  so the benchmark table reads as "what the network adds to textbook physics".

`ModelConfig.physics_residual` turns this off for the ablation.
"""

from __future__ import annotations

import numpy as np

from ..geometry.features import FEATURE_NAMES
from .base import VesselContext

_LOG_RADIUS_RATIO = FEATURE_NAMES.index("log_radius_ratio")


def local_radius(features: np.ndarray, context: VesselContext) -> np.ndarray:
    """Recover absolute local radius in metres from the dimensionless feature."""
    log_ratio = np.asarray(features[..., _LOG_RADIUS_RATIO], dtype=np.float64)
    inlet = np.asarray(context.inlet_radius, dtype=np.float64)
    if log_ratio.ndim == 3:
        inlet = inlet[:, None, None]
    return np.maximum(inlet * np.exp(log_ratio), 1e-9)


def poiseuille_field(features: np.ndarray, context: VesselContext) -> np.ndarray:
    """The analytic prior `4 mu Q / (pi r^3)` on the full surface grid.

    Args:
        features: `(..., n_arc, n_theta, n_features)` raw features.
        context: physical scale of the same vessels.

    Returns:
        Wall shear stress in pascals, same leading shape as the feature grid.
    """
    radius = local_radius(features, context)
    flow = np.asarray(context.flow_rate, dtype=np.float64)
    if radius.ndim == 3:
        flow = flow[:, None, None]
    return 4.0 * context.viscosity * flow / (np.pi * radius**3)
