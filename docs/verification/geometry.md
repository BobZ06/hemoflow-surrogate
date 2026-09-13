# Geometry: vessel representation and the 12 features

**Owner:** Bowen. Verifies `src/hemoflow/geometry/vessel.py`, `geometry/features.py`.

## Parallel transport vs. a Frenet frame

A Frenet frame is built from the curvature vector, so it flips 180 degrees
discontinuously wherever curvature passes through zero - which on an
almost-straight vessel happens from numerical noise alone. That flip would cut
an artificial seam through the circumferential coordinate at a point where the
geometry itself is perfectly smooth.

`parallel_transport_frames` instead advances the frame by the minimal rotation
between consecutive tangents, and where the tangent barely moves
(`sin_a < 1e-9`) it carries the previous frame through unchanged. No flip, no
seam, even on a perfectly straight segment.

## The part that is easy to miss

Parallel transport fixes the frame's *twist* but not its *starting angle*: the
seed vector is picked from a world axis, so rotating the whole vessel shifts
every phase by one constant offset. Left alone, that offset moves where the
discrete theta samples land relative to the bend, and the extracted features
change slightly under rotation - invariance would hold for the continuous field
but not for the grid the model actually sees.

`build_vessel` cancels it by subtracting a curvature-weighted circular mean of
the phase, which shifts with the frame. On a straight vessel the weights vanish
and the anchor is 0, which is harmless: with no bend there is no preferred
direction to align to.

## The 12 features (`FEATURE_NAMES`)

1. `log_radius_ratio` - local radius vs inlet radius, log scale
2. `dradius_ds` - converging (<0) or diverging (>0) wall
3. `d2radius_ds2` - curvature of the radius profile
4. `curvature_x_radius` - dimensionless centerline bend
5. `arc_fraction` - normalised position along the vessel
6. `cos_rel_theta`
7. `sin_rel_theta` - circumferential angle from the bend's inner wall
8. `log_reynolds` - local Reynolds number
9. `log_dean` - secondary-flow strength in bends
10. `area_ratio` - local area vs inlet area
11. `throat_ratio` - min upstream radius vs local radius
12. `downstream_of_throat` - diameters travelled since the global minimum

## Hand-traced: a single stenosis at the midpoint

- `throat_ratio` = running-min(radius) / radius(here). Upstream of the throat
  the radius falls monotonically, so the running minimum tracks the current
  radius and the ratio sits at ~1.0 the whole way in. Past the throat the
  radius climbs back but the running minimum stays pinned at the throat value,
  so the ratio drops below 1 and keeps falling downstream.
- `downstream_of_throat` = (arc - arc_at_throat) / diameter_at_throat, clipped
  to [-20, 20]: negative upstream, exactly 0 at the throat, a straight ramp
  after - measured in throat-diameters, so it means the same thing on a wide
  vessel and a narrow one.

These two are why the U-Net buys almost nothing over the per-node MLP: the
non-local information a convolution would have to discover is already in the
feature vector. See `docs/verification/evaluation.md`.

## Verified

- [x] Ran `pytest tests/test_geometry.py -v`: **8 passed**
- [x] `test_features_are_rotation_invariant` is the one that matters - rotate a
      vessel, re-extract, features must come back identical. It would catch any
      feature computed from a raw coordinate instead of the invariant frame,
      which would train fine and then fail on a patient scanned at a different
      angle.
- [x] Traced `throat_ratio` and `downstream_of_throat` by hand, above
