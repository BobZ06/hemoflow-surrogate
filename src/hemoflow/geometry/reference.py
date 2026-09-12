"""Reference wall-shear-stress solver - the thing the surrogate is trained to replace.

**Read this before trusting any number this repo prints.**

A production surrogate is trained on a full 3D Navier-Stokes solve: hours of
CPU per geometry, produced offline by a solver such as SimVascular or OpenFOAM.
That is exactly the cost the surrogate exists to amortise, and it is exactly the
cost that does not fit inside a 24-hour build.

So this module implements a *reduced-order* reference instead: a Poiseuille core
with physically motivated corrections for convective acceleration, post-stenotic
separation, entrance development, and Dean-type secondary flow in curved
segments. It reproduces the qualitative structure of a real WSS field - axial
peaks at throats, a low-shear recirculation zone downstream, and a
circumferential high on the outer wall of a bend - at microseconds per case.

The contract that matters: `reference_wss` is an *interface*, not a commitment.
Everything downstream (features, dataset, models, metrics, service) consumes its
output and nothing downstream knows how it was produced. Swapping in real solver
output means replacing this one function and regenerating the dataset; no other
module changes. `docs/ARCHITECTURE.md` spells out that swap.

Accordingly, every metric produced against this reference measures *fidelity to
a reduced-order model*, not clinical accuracy. Do not present it as the latter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FluidConfig
from .vessel import Vessel


@dataclass(frozen=True)
class ReferenceCoefficients:
    """Tunable coefficients of the reduced-order model.

    These are shape parameters chosen to put each effect in a physiologically
    sensible range, not values fitted to data. They are grouped here so that the
    reference model has no magic numbers buried in the code path.
    """

    convergence_gain: float = 2.0
    separation_gain: float = 6.0
    separation_length_diameters: float = 2.5
    entrance_gain: float = 0.45
    entrance_length_diameters: float = 4.0
    dean_gain: float = 0.38
    dean_reference: float = 90.0
    floor_fraction: float = 0.03


def _causal_ema(signal: np.ndarray, arc_length: np.ndarray, length_scale: np.ndarray) -> np.ndarray:
    """Downstream-only exponential memory along the centerline.

    Separation is history-dependent: shear downstream of a throat is depressed
    because of what the flow did upstream, and it recovers over a few diameters.
    A causal filter is the cheapest honest way to express that; a pointwise
    function of local radius cannot, which is precisely why the Poiseuille
    baseline has an irreducible error on these fields.
    """
    out = np.zeros_like(signal)
    out[0] = signal[0]
    steps = np.diff(arc_length)
    for i in range(1, len(signal)):
        scale = max(float(length_scale[i]), 1e-6)
        alpha = float(np.exp(-steps[i - 1] / scale))
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * signal[i]
    return out


def poiseuille_wss(radius: np.ndarray, flow_rate: float, viscosity: float) -> np.ndarray:
    """Analytic WSS for steady laminar flow in a straight circular pipe.

    `tau = 4 * mu * Q / (pi * r^3)`

    The cubic dependence on radius is why stenosis severity dominates the whole
    problem: a 30% narrowing nearly triples wall shear stress.
    """
    radius = np.maximum(np.asarray(radius, dtype=np.float64), 1e-6)
    return 4.0 * viscosity * float(flow_rate) / (np.pi * radius**3)


def reference_wss(
    vessel: Vessel,
    fluid: FluidConfig,
    coeffs: ReferenceCoefficients | None = None,
) -> np.ndarray:
    """Compute the reference WSS field on the vessel surface grid.

    Args:
        vessel: geometry to solve on.
        fluid: blood properties and nothing else.
        coeffs: optional override of the reduced-order coefficients.

    Returns:
        `(n_arc, n_theta)` wall shear stress in pascals. Physiological arterial
        values sit roughly in 0.5-7 Pa; throats run higher and recirculation
        zones lower.
    """
    coeffs = coeffs or ReferenceCoefficients()
    mu, rho = fluid.viscosity_pa_s, fluid.density_kg_m3
    radius, arc = vessel.radius, vessel.arc_length
    diameter = 2.0 * radius

    # --- axial core ---------------------------------------------------------
    tau_0 = poiseuille_wss(radius, vessel.flow_rate, mu)

    velocity = vessel.flow_rate / (np.pi * np.maximum(radius, 1e-6) ** 2)
    reynolds = rho * velocity * diameter / mu

    # --- convective acceleration and post-stenotic separation ---------------
    dr_ds = np.gradient(radius, arc)
    convergence = np.clip(-dr_ds, 0.0, None)
    divergence = np.clip(dr_ds, 0.0, None)

    f_converge = 1.0 + coeffs.convergence_gain * convergence
    wake = _causal_ema(divergence, arc, coeffs.separation_length_diameters * diameter)
    f_separate = 1.0 / (1.0 + coeffs.separation_gain * wake)

    # --- entrance development ------------------------------------------------
    entrance_scale = max(coeffs.entrance_length_diameters * float(diameter[0]), 1e-6)
    f_entrance = 1.0 + coeffs.entrance_gain * np.exp(-arc / entrance_scale)

    axial = tau_0 * f_converge * f_separate * f_entrance

    # --- Dean secondary flow: circumferential asymmetry in bends -------------
    # De = Re * sqrt(r / R_curvature), and R_curvature = 1 / kappa.
    dean = reynolds * np.sqrt(np.maximum(radius * vessel.curvature, 0.0))
    asymmetry = coeffs.dean_gain * np.tanh(dean / coeffs.dean_reference)
    # theta measured from the inner wall of the bend, so -cos() peaks outside.
    rel_theta = vessel.relative_theta()
    f_dean = 1.0 + asymmetry[:, None] * (-np.cos(rel_theta))

    field = axial[:, None] * f_dean

    floor = coeffs.floor_fraction * float(np.mean(tau_0))
    return np.maximum(field, floor)
