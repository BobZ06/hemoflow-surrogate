"""Non-neural baselines. Pure numpy, no training loop, no GPU.

These exist so that the neural results mean something. A surrogate that beats
nothing has not been shown to work; a surrogate that beats closed-form physics
and a linear model on the same split has.

Three rungs, in increasing order of what they are allowed to know:

* `ConstantBaseline` - predicts the training mean. Any model that fails to beat
  this is broken, not merely weak.
* `PoiseuilleBaseline` - textbook physics, zero fitted parameters. Captures the
  dominant `1/r^3` behaviour and nothing else: it is blind to separation,
  entrance effects and every circumferential variation, because the analytic
  solution has no angular dependence at all.
* `RidgeBaseline` - closed-form linear fit on the same features the networks
  see. Separates "the features are informative" from "the architecture is doing
  something".
"""

from __future__ import annotations

import numpy as np

from ..geometry.features import FEATURE_NAMES
from .base import Surrogate, VesselContext

_LOG_RADIUS_RATIO = FEATURE_NAMES.index("log_radius_ratio")


class ConstantBaseline(Surrogate):
    """Predicts one number everywhere: the geometric mean of the training field."""

    name = "constant"

    def __init__(self, value_pa: float) -> None:
        self.value_pa = float(value_pa)

    @classmethod
    def fit(cls, targets: np.ndarray) -> ConstantBaseline:
        # Geometric mean, because the error metric is relative and the target is
        # log-distributed; the arithmetic mean would sit far above the median.
        return cls(float(np.exp(np.mean(np.log(np.maximum(targets, 1e-8))))))

    def predict_pa(self, features: np.ndarray, context: VesselContext) -> np.ndarray:
        return np.full(features.shape[:3], self.value_pa, dtype=np.float64)


class PoiseuilleBaseline(Surrogate):
    """Closed-form WSS for fully developed laminar flow: `4 mu Q / (pi r^3)`.

    Zero fitted parameters. Local radius is recovered from the dimensionless
    feature and the inlet radius carried in the context.

    Its failure mode is the interesting part: being a pointwise function of
    radius, it is constant around the circumference, so its error on a curved
    vessel is a *lower bound* on the value of any model that can represent
    angular variation.
    """

    name = "poiseuille"

    def predict_pa(self, features: np.ndarray, context: VesselContext) -> np.ndarray:
        log_ratio = np.asarray(features[..., _LOG_RADIUS_RATIO], dtype=np.float64)
        radius = context.inlet_radius[:, None, None] * np.exp(log_ratio)
        radius = np.maximum(radius, 1e-6)
        flow = np.asarray(context.flow_rate, dtype=np.float64)[:, None, None]
        return 4.0 * context.viscosity * flow / (np.pi * radius**3)


class RidgeBaseline(Surrogate):
    """Ridge regression from node features to log-WSS, solved in closed form.

    Fitted per node with no spatial structure at all, which is exactly the point:
    whatever a convolutional or graph model gains over this number is the value
    of modelling the neighbourhood rather than the point.
    """

    name = "ridge"

    def __init__(self, weights: np.ndarray, bias: float) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)

    @property
    def n_parameters(self) -> int:
        return int(self.weights.size) + 1

    @classmethod
    def fit(cls, features: np.ndarray, targets: np.ndarray, alpha: float = 1.0) -> RidgeBaseline:
        """Least squares with L2 regularisation, on the training split only.

        Args:
            features: `(n_vessels, n_arc, n_theta, n_features)` raw features.
            targets: `(n_vessels, n_arc, n_theta)` WSS in pascals.
            alpha: ridge penalty.
        """
        x = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
        y = np.log(np.maximum(np.asarray(targets, dtype=np.float64).reshape(-1), 1e-8))

        x_mean, y_mean = x.mean(axis=0), y.mean()
        xc, yc = x - x_mean, y - y_mean

        gram = xc.T @ xc + alpha * np.eye(x.shape[1])
        weights = np.linalg.solve(gram, xc.T @ yc)
        bias = y_mean - float(x_mean @ weights)
        return cls(weights=weights, bias=bias)

    def predict_pa(self, features: np.ndarray, context: VesselContext) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        log_pred = x @ self.weights + self.bias
        return np.exp(np.clip(log_pred, -30.0, 30.0))
