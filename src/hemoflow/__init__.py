"""hemoflow - a neural surrogate for hemodynamic wall shear stress.

The package is organised as a straight line through the problem:

    geometry  ->  data  ->  models  ->  training  ->  serving

`geometry` turns a vessel into rotation-invariant arrays, `data` turns many
vessels into a leakage-safe tensor dataset, `models` maps those arrays to a
wall-shear-stress field, `training` fits and scores them, and `serving`
exposes the fitted model behind an HTTP endpoint with a latency budget.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
