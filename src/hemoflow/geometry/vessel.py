"""Vessel geometry: centerline, radius profile, and a stable moving frame.

The single most important decision in this repo is how a 3D vessel is handed to
a neural network. Feeding raw xyz coordinates forces the model to spend its
capacity learning that a rotated artery is the same artery. Instead every
vessel is reduced to a centerline plus a radius field sampled on an
`(arc x circumference)` grid, and all features are expressed relative to a
*parallel-transport frame* carried along that centerline.

That frame is what makes the representation rotation-invariant by construction:
rotate the whole vessel in space and every feature this module produces is
numerically unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, EPS)


def _rotate_about_axis(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of a single 3-vector."""
    axis = axis / max(float(np.linalg.norm(axis)), EPS)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return vec * cos_a + np.cross(axis, vec) * sin_a + axis * np.dot(axis, vec) * (1.0 - cos_a)


def parallel_transport_frames(tangents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Carry an orthonormal frame along the centerline without twisting it.

    A Frenet frame flips discontinuously wherever curvature passes through
    zero, which would put an artificial seam in the circumferential coordinate
    of an almost-straight vessel. Parallel transport advances the frame by the
    minimal rotation between consecutive tangents instead, so it stays smooth
    even on perfectly straight segments.

    Args:
        tangents: `(n_arc, 3)` unit tangents along the centerline.

    Returns:
        `(u, v)`, each `(n_arc, 3)`, forming a right-handed orthonormal frame
        with the tangent at every station.
    """
    tangents = _normalize(np.asarray(tangents, dtype=np.float64))
    n = len(tangents)

    # Seed with any vector not parallel to the first tangent.
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(seed, tangents[0]))) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    u0 = _normalize(seed - np.dot(seed, tangents[0]) * tangents[0])

    u = np.zeros((n, 3))
    u[0] = u0
    for i in range(1, n):
        prev_t, cur_t = tangents[i - 1], tangents[i]
        axis = np.cross(prev_t, cur_t)
        sin_a = float(np.linalg.norm(axis))
        cos_a = float(np.clip(np.dot(prev_t, cur_t), -1.0, 1.0))
        if sin_a < 1e-9:
            # Tangent barely moved: carry the previous frame through unchanged.
            candidate = u[i - 1]
        else:
            candidate = _rotate_about_axis(u[i - 1], axis, float(np.arctan2(sin_a, cos_a)))
        # Re-orthogonalise against drift accumulated by repeated rotation.
        candidate = candidate - np.dot(candidate, cur_t) * cur_t
        u[i] = _normalize(candidate[None, :])[0]

    v = np.cross(tangents, u)
    return u, _normalize(v)


@dataclass(frozen=True)
class Vessel:
    """One vessel sample discretised on an `(n_arc, n_theta)` surface grid.

    Attributes:
        centerline: `(n_arc, 3)` centerline positions in metres.
        arc_length: `(n_arc,)` cumulative arc length from the inlet, in metres.
        radius: `(n_arc,)` local lumen radius in metres.
        tangent: `(n_arc, 3)` unit tangents.
        frame_u, frame_v: `(n_arc, 3)` parallel-transport frame vectors.
        curvature: `(n_arc,)` centerline curvature magnitude, 1/m.
        curvature_phase: `(n_arc,)` angle of the curvature vector within the
            `(u, v)` frame. Subtracting it from the circumferential coordinate
            yields an angle measured from the *inner wall of the bend*, which is
            the physically meaningful reference direction.
        theta: `(n_theta,)` circumferential grid in radians.
        flow_rate: volumetric inflow in m^3/s.
        vessel_id: stable identifier used to keep splits leakage-free.
    """

    centerline: np.ndarray
    arc_length: np.ndarray
    radius: np.ndarray
    tangent: np.ndarray
    frame_u: np.ndarray
    frame_v: np.ndarray
    curvature: np.ndarray
    curvature_phase: np.ndarray
    theta: np.ndarray
    flow_rate: float
    vessel_id: str

    @property
    def n_arc(self) -> int:
        return len(self.arc_length)

    @property
    def n_theta(self) -> int:
        return len(self.theta)

    @property
    def length(self) -> float:
        return float(self.arc_length[-1])

    def surface_points(self) -> np.ndarray:
        """Materialise the `(n_arc, n_theta, 3)` surface point cloud.

        Only needed for rendering and for mesh export. The model never sees
        these coordinates, which is the entire point of the representation.
        """
        cos_t = np.cos(self.theta)[None, :, None]
        sin_t = np.sin(self.theta)[None, :, None]
        radial = cos_t * self.frame_u[:, None, :] + sin_t * self.frame_v[:, None, :]
        return self.centerline[:, None, :] + self.radius[:, None, None] * radial

    def relative_theta(self) -> np.ndarray:
        """`(n_arc, n_theta)` circumferential angle measured from the bend's inner wall.

        Wrapped to `[-pi, pi)`. This is the coordinate the model consumes.
        """
        rel = self.theta[None, :] - self.curvature_phase[:, None]
        return (rel + np.pi) % (2.0 * np.pi) - np.pi


def build_vessel(
    centerline: np.ndarray,
    radius: np.ndarray,
    n_theta: int,
    flow_rate: float,
    vessel_id: str,
) -> Vessel:
    """Assemble a `Vessel` from a centerline and radius profile.

    Derivatives are taken with respect to true arc length rather than the index,
    so a vessel remains the same vessel under resampling.
    """
    centerline = np.asarray(centerline, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)

    steps = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    arc_length = np.concatenate([[0.0], np.cumsum(steps)])

    tangent = _normalize(np.gradient(centerline, arc_length, axis=0))
    dt_ds = np.gradient(tangent, arc_length, axis=0)
    # Only the component orthogonal to the tangent is curvature.
    dt_ds = dt_ds - np.sum(dt_ds * tangent, axis=1, keepdims=True) * tangent
    curvature = np.linalg.norm(dt_ds, axis=1)

    frame_u, frame_v = parallel_transport_frames(tangent)
    curvature_phase = np.arctan2(np.sum(dt_ds * frame_v, axis=1), np.sum(dt_ds * frame_u, axis=1))

    # Anchor the circumferential grid to the vessel's own bend.
    #
    # Parallel transport fixes the frame's *twist* but not its starting angle:
    # the seed vector is picked from a world axis, so rotating the whole vessel
    # shifts every phase by one constant offset. Left alone, that offset moves
    # where the discrete theta samples land relative to the bend, and the
    # extracted features change slightly under rotation - the invariance would
    # hold for the continuous field but not for the grid the model actually sees.
    #
    # Subtracting a curvature-weighted circular mean of the phase cancels the
    # offset exactly, because it shifts with the frame. On a straight vessel the
    # weights vanish and the anchor is 0, which is harmless: with no bend there
    # is no preferred direction to align to.
    weight = curvature
    anchor = float(
        np.arctan2(
            np.sum(weight * np.sin(curvature_phase)),
            np.sum(weight * np.cos(curvature_phase)),
        )
    )
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False) + anchor

    return Vessel(
        centerline=centerline,
        arc_length=arc_length,
        radius=radius,
        tangent=tangent,
        frame_u=frame_u,
        frame_v=frame_v,
        curvature=curvature,
        curvature_phase=curvature_phase,
        theta=theta,
        flow_rate=float(flow_rate),
        vessel_id=vessel_id,
    )


def vessel_from_profile(
    radii: np.ndarray,
    length: float,
    curvature: float,
    n_theta: int,
    flow_rate: float,
    vessel_id: str = "request",
) -> Vessel:
    """Build a vessel from a compact description: radius profile, length, one bend.

    This is the shape the inference API accepts. A full centerline is the honest
    input for a real case, but a planar arc of constant curvature plus a radius
    profile captures the two factors that dominate wall shear stress, and it is
    something a caller can type by hand or drive from a slider.

    Args:
        radii: `(n_arc,)` lumen radius in metres, inlet first.
        length: centerline arc length in metres.
        curvature: constant centerline curvature in 1/m; 0 is a straight vessel.
        n_theta: circumferential resolution.
        flow_rate: volumetric inflow in m^3/s.
        vessel_id: identifier echoed back in responses.
    """
    radii = np.asarray(radii, dtype=np.float64)
    n = len(radii)
    if n < 4:
        raise ValueError("need at least 4 radius samples to differentiate the profile")
    if length <= 0:
        raise ValueError("length must be positive")

    s = np.linspace(0.0, float(length), n)
    k = float(curvature)
    if abs(k) < 1e-9:
        centerline = np.stack([np.zeros(n), np.zeros(n), s], axis=1)
    else:
        radius_of_curvature = 1.0 / k
        angle = s * k
        centerline = np.stack(
            [
                radius_of_curvature * (1.0 - np.cos(angle)),
                np.zeros(n),
                radius_of_curvature * np.sin(angle),
            ],
            axis=1,
        )

    return build_vessel(
        centerline=centerline,
        radius=radii,
        n_theta=n_theta,
        flow_rate=float(flow_rate),
        vessel_id=vessel_id,
    )
