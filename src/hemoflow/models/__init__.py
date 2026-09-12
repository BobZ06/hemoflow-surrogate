"""The model zoo: analytic baselines, a linear control, and two networks.

Baselines are numpy-only and import eagerly. Anything that needs torch is
resolved lazily, so `hemoflow data` and the baseline comparison still run on a
machine without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Surrogate, VesselContext
from .baselines import ConstantBaseline, PoiseuilleBaseline, RidgeBaseline
from .physics import local_radius, poiseuille_field

if TYPE_CHECKING:  # pragma: no cover
    from .nets import NodeMLP, VesselUNet, build_network
    from .registry import TorchSurrogate, load_surrogate, resolve_device, save_checkpoint

_LAZY = {
    "NodeMLP": "nets",
    "VesselUNet": "nets",
    "build_network": "nets",
    "TorchSurrogate": "registry",
    "load_surrogate": "registry",
    "resolve_device": "registry",
    "save_checkpoint": "registry",
}

__all__ = [
    "ConstantBaseline",
    "NodeMLP",
    "PoiseuilleBaseline",
    "RidgeBaseline",
    "Surrogate",
    "TorchSurrogate",
    "VesselContext",
    "VesselUNet",
    "build_network",
    "local_radius",
    "poiseuille_field",
    "load_surrogate",
    "resolve_device",
    "save_checkpoint",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)
