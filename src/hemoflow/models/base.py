"""The one interface every surrogate implements.

An analytic formula, a ridge regression and a U-Net have nothing in common
internally, but the evaluation harness should not care. They all answer the same
question - *given this geometry and this inflow, what is the wall shear stress
at every surface node, in pascals* - so they all implement `predict_pa`.

Returning physical units from every model, rather than whatever normalised space
each happens to train in, is what makes the benchmark table in the README an
apples-to-apples comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VesselContext:
    """Physical scale of a batch of vessels.

    The feature vector is deliberately dimensionless so that the model
    generalises across vessel calibres. That means absolute scale has to travel
    separately, and this is it.

    Attributes:
        flow_rate: `(n_vessels,)` volumetric inflow, m^3/s.
        inlet_radius: `(n_vessels,)` radius at the inlet, m.
        viscosity: dynamic viscosity of blood, Pa*s.
    """

    flow_rate: np.ndarray
    inlet_radius: np.ndarray
    viscosity: float = 0.0035

    def __len__(self) -> int:
        return len(self.flow_rate)


class Surrogate(ABC):
    """A model that maps vessel features to a wall-shear-stress field."""

    name: str = "surrogate"

    @abstractmethod
    def predict_pa(self, features: np.ndarray, context: VesselContext) -> np.ndarray:
        """Predict wall shear stress in pascals.

        Args:
            features: `(n_vessels, n_arc, n_theta, n_features)` *raw* features,
                exactly as produced by `geometry.extract_features`. Any
                normalisation a model needs is the model's own business - a
                caller must never have to remember to scale inputs first, since
                forgetting is the single most common train/serve skew bug.
            context: physical scale of each vessel.

        Returns:
            `(n_vessels, n_arc, n_theta)` wall shear stress in pascals.
        """

    @property
    def n_parameters(self) -> int:
        """Learnable parameter count, reported in the benchmark table."""
        return 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, params={self.n_parameters})"
