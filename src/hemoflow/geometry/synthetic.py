"""Synthetic vessel sampler.

Real patient geometries are scarce and arrive late. This module generates an
unlimited supply of anatomically plausible vessels so the pipeline, the
training loop, the metrics and the service can all be built and validated
before a single byte of clinical data lands.

The distribution is deliberately broad in the two factors that dominate wall
shear stress: *lumen narrowing* and *centerline curvature*. Everything else is
mild smooth variation.
"""

from __future__ import annotations

import numpy as np

from ..config import FluidConfig, GeometryConfig
from .vessel import Vessel, build_vessel


def _smooth_noise(rng: np.random.Generator, n: int, n_modes: int, scale: float) -> np.ndarray:
    """Band-limited random signal: a few low-frequency Fourier modes.

    Band-limiting matters. White noise on the radius would create geometry that
    no vessel has and that no mesher could resolve, and the model would waste
    capacity on it.
    """
    t = np.linspace(0.0, 1.0, n)
    signal = np.zeros(n)
    for mode in range(1, n_modes + 1):
        amplitude = rng.normal(0.0, scale / mode**2)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal += amplitude * np.sin(2.0 * np.pi * mode * t + phase)
    return signal


def sample_vessel(
    rng: np.random.Generator,
    geometry: GeometryConfig,
    fluid: FluidConfig,
    vessel_id: str,
) -> Vessel:
    """Draw one random vessel.

    Args:
        rng: seeded generator; the only source of randomness.
        geometry: discretisation and shape ranges.
        fluid: used solely to sample an inflow rate in a physiological band.
        vessel_id: stable id, carried into the dataset so that splits can be
            made by vessel rather than by surface node.

    Returns:
        A fully built `Vessel`.
    """
    n = geometry.n_arc
    s_norm = np.linspace(0.0, 1.0, n)

    # --- centerline: a gently curving tube through space ---------------------
    axis = s_norm * geometry.length_m
    off_x = _smooth_noise(rng, n, n_modes=3, scale=geometry.curvature_scale)
    off_y = _smooth_noise(rng, n, n_modes=3, scale=geometry.curvature_scale)
    # Pin the inlet so curvature is not dominated by an arbitrary offset.
    off_x -= off_x[0]
    off_y -= off_y[0]
    centerline = np.stack([off_x, off_y, axis], axis=1)

    # --- radius: baseline taper, smooth variation, and one focal stenosis ----
    base = geometry.base_radius_m * rng.uniform(0.75, 1.25)
    taper = np.linspace(1.0, rng.uniform(0.82, 1.0), n)
    wobble = 1.0 + _smooth_noise(rng, n, n_modes=2, scale=geometry.radius_jitter * 0.5)
    radius = base * taper * np.clip(wobble, 0.7, 1.3)

    severity = float(rng.uniform(0.0, geometry.max_stenosis))
    if severity > 0.05:
        center = float(rng.uniform(0.25, 0.75))
        width = float(rng.uniform(0.06, 0.15))
        bump = np.exp(-0.5 * ((s_norm - center) / width) ** 2)
        radius = radius * (1.0 - severity * bump)

    radius = np.maximum(radius, 0.15 * base)

    flow_rate = float(rng.uniform(fluid.flow_min_m3_s, fluid.flow_max_m3_s))

    return build_vessel(
        centerline=centerline,
        radius=radius,
        n_theta=geometry.n_theta,
        flow_rate=flow_rate,
        vessel_id=vessel_id,
    )
