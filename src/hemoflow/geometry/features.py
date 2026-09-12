"""Feature extraction: vessel geometry -> rotation-invariant node features.

Every feature here is a scalar that survives an arbitrary rigid motion of the
vessel. There is no raw coordinate anywhere in the vector. That is a deliberate
architectural choice with a measurable payoff: an SE(3)-equivariant network
would have to *learn* this invariance from data, and with a few hundred training
geometries it will not. Building the invariance into the input instead spends
zero parameters on it.

`test_geometry.py` enforces the property directly - rotate a vessel, re-extract,
assert the features are bit-for-bit comparable.
"""

from __future__ import annotations

import numpy as np

from ..config import FluidConfig
from .vessel import Vessel

FEATURE_NAMES: tuple[str, ...] = (
    "log_radius_ratio",  # local radius vs. inlet radius, log scale
    "dradius_ds",  # converging (<0) or diverging (>0) wall
    "d2radius_ds2",  # curvature of the radius profile
    "curvature_x_radius",  # dimensionless centerline bend
    "arc_fraction",  # normalised position along the vessel
    "cos_rel_theta",  # circumferential angle from the bend's inner wall
    "sin_rel_theta",
    "log_reynolds",  # local Reynolds number
    "log_dean",  # secondary-flow strength in bends
    "area_ratio",  # local area vs. inlet area
    "throat_ratio",  # min upstream radius vs. local radius
    "downstream_of_throat",  # diameters travelled since the global minimum
)

N_FEATURES = len(FEATURE_NAMES)


def extract_features(vessel: Vessel, fluid: FluidConfig) -> np.ndarray:
    """Build the `(n_arc, n_theta, N_FEATURES)` feature tensor for one vessel.

    Args:
        vessel: the geometry.
        fluid: blood properties, used for Reynolds and Dean numbers.

    Returns:
        Float32 array of node features, ordered as `FEATURE_NAMES`.

    Note:
        Uses only geometry and inflow. Nothing derived from the reference
        solution appears here, so the same code path runs unchanged at training
        time and behind the inference API.
    """
    radius = vessel.radius
    arc = vessel.arc_length
    inlet_radius = float(radius[0])
    diameter = 2.0 * radius

    dr_ds = np.gradient(radius, arc)
    d2r_ds2 = np.gradient(dr_ds, arc)

    velocity = vessel.flow_rate / (np.pi * np.maximum(radius, 1e-6) ** 2)
    reynolds = fluid.density_kg_m3 * velocity * diameter / fluid.viscosity_pa_s
    dean = reynolds * np.sqrt(np.maximum(radius * vessel.curvature, 0.0))

    # Running minimum radius upstream of each station: how much narrowing the
    # flow has already been through when it arrives here.
    upstream_min = np.minimum.accumulate(radius)

    throat_index = int(np.argmin(radius))
    distance_from_throat = (arc - arc[throat_index]) / max(float(diameter[throat_index]), 1e-6)

    rel_theta = vessel.relative_theta()
    n_theta = vessel.n_theta

    def along(values: np.ndarray) -> np.ndarray:
        """Broadcast a per-station quantity across the circumference."""
        return np.repeat(np.asarray(values, dtype=np.float64)[:, None], n_theta, axis=1)

    channels = [
        along(np.log(np.maximum(radius / max(inlet_radius, 1e-9), 1e-6))),
        along(dr_ds),
        along(d2r_ds2 * inlet_radius),  # scaled to keep magnitudes comparable
        along(vessel.curvature * radius),
        along(arc / max(float(arc[-1]), 1e-9)),
        np.cos(rel_theta),
        np.sin(rel_theta),
        along(np.log1p(reynolds)),
        along(np.log1p(dean)),
        along((radius / max(inlet_radius, 1e-9)) ** 2),
        along(upstream_min / np.maximum(radius, 1e-9)),
        along(np.clip(distance_from_throat, -20.0, 20.0)),
    ]

    stacked = np.stack(channels, axis=-1)
    assert stacked.shape[-1] == N_FEATURES, "channel list and FEATURE_NAMES disagree"
    return stacked.astype(np.float32)
