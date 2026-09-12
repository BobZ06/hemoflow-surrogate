"""Geometry invariants.

The headline test is `test_features_are_rotation_invariant`. The whole
representation argument in `geometry/vessel.py` rests on that property, and a
property that is only argued for in a docstring is a property that will quietly
stop holding the first time someone edits the feature extractor.
"""

from __future__ import annotations

import numpy as np
import pytest

from hemoflow.config import FluidConfig, GeometryConfig
from hemoflow.geometry import (
    build_vessel,
    extract_features,
    parallel_transport_frames,
    reference_wss,
    sample_vessel,
    vessel_from_profile,
)
from hemoflow.geometry.features import N_FEATURES


@pytest.fixture
def geometry() -> GeometryConfig:
    return GeometryConfig(n_arc=64, n_theta=32)


@pytest.fixture
def fluid() -> FluidConfig:
    return FluidConfig()


def _random_rotation(seed: int) -> np.ndarray:
    """A uniformly random rotation matrix via QR of a Gaussian matrix."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_parallel_transport_frame_is_orthonormal(geometry, fluid):
    vessel = sample_vessel(np.random.default_rng(0), geometry, fluid, "v")
    u, v, t = vessel.frame_u, vessel.frame_v, vessel.tangent

    assert np.allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-9)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-9)
    assert np.allclose(np.sum(u * t, axis=1), 0.0, atol=1e-9)
    assert np.allclose(np.sum(v * t, axis=1), 0.0, atol=1e-9)
    assert np.allclose(np.sum(u * v, axis=1), 0.0, atol=1e-9)


def test_parallel_transport_does_not_twist_on_a_straight_line():
    """A straight centerline has no curvature, so the frame must not rotate."""
    tangents = np.tile(np.array([0.0, 0.0, 1.0]), (32, 1))
    u, v = parallel_transport_frames(tangents)
    assert np.allclose(u, u[0], atol=1e-12)
    assert np.allclose(v, v[0], atol=1e-12)


def test_features_are_rotation_invariant(geometry, fluid):
    """Rotating a vessel in space must not change a single feature value.

    This is the property that lets the model spend all of its capacity on
    hemodynamics instead of on learning that orientation is irrelevant.
    """
    vessel = sample_vessel(np.random.default_rng(7), geometry, fluid, "v")
    original = extract_features(vessel, fluid)

    rotation = _random_rotation(11)
    translation = np.array([0.3, -1.2, 5.0])
    rotated = build_vessel(
        centerline=vessel.centerline @ rotation.T + translation,
        radius=vessel.radius,
        n_theta=vessel.n_theta,
        flow_rate=vessel.flow_rate,
        vessel_id="rotated",
    )
    transformed = extract_features(rotated, fluid)

    # Element-wise, not merely as a set: because the circumferential grid is
    # anchored to the vessel's own bend, node (i, j) of the rotated vessel is the
    # same physical point as node (i, j) of the original.
    np.testing.assert_allclose(original, transformed, rtol=1e-4, atol=1e-5)


def test_reference_field_is_rotation_invariant(geometry, fluid):
    vessel = sample_vessel(np.random.default_rng(3), geometry, fluid, "v")
    rotation = _random_rotation(5)
    rotated = build_vessel(
        centerline=vessel.centerline @ rotation.T,
        radius=vessel.radius,
        n_theta=vessel.n_theta,
        flow_rate=vessel.flow_rate,
        vessel_id="rotated",
    )
    a = reference_wss(vessel, fluid)
    b = reference_wss(rotated, fluid)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_feature_tensor_shape_and_finiteness(geometry, fluid):
    vessel = sample_vessel(np.random.default_rng(1), geometry, fluid, "v")
    features = extract_features(vessel, fluid)
    assert features.shape == (geometry.n_arc, geometry.n_theta, N_FEATURES)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_narrowing_raises_wall_shear_stress(fluid):
    """WSS scales as 1/r^3, so a throat must carry more shear than the inlet."""
    n = 64
    s = np.linspace(0.0, 1.0, n)
    radii = 0.0025 * (1.0 - 0.5 * np.exp(-0.5 * ((s - 0.5) / 0.08) ** 2))
    vessel = vessel_from_profile(radii, length=0.06, curvature=0.0, n_theta=16, flow_rate=4e-6)
    field = reference_wss(vessel, fluid)

    throat = int(np.argmin(radii))
    assert field[throat].mean() > 3.0 * field[0].mean()


def test_curvature_creates_circumferential_variation(fluid):
    """A bend must break the angular symmetry that a straight vessel has."""
    radii = np.full(64, 0.0025)
    straight = vessel_from_profile(radii, 0.06, 0.0, 32, 6e-6)
    curved = vessel_from_profile(radii, 0.06, 40.0, 32, 6e-6)

    straight_spread = np.ptp(reference_wss(straight, fluid), axis=1).mean()
    curved_spread = np.ptp(reference_wss(curved, fluid), axis=1).mean()

    assert straight_spread < 1e-9, "a straight vessel has no preferred direction"
    assert curved_spread > 0.05, "a bend must raise shear on the outer wall"


def test_vessel_from_profile_rejects_bad_input():
    with pytest.raises(ValueError):
        vessel_from_profile(np.array([0.002, 0.002]), 0.06, 0.0, 16, 4e-6)
    with pytest.raises(ValueError):
        vessel_from_profile(np.full(16, 0.002), -1.0, 0.0, 16, 4e-6)
