# Working on this repo

Two people, one weekend, one `main` branch that always works. The rules below
exist to keep us from stepping on each other, not to be ceremonial.

## Setup

```bash
git clone <this repo>
cd hemoflow-surrogate
bash scripts/setup.sh     # installs the CPU torch build + this package
make smoke                # ~40s end-to-end sanity check
```

If `make smoke` fails on a clean clone, that is a bug and it is the most
important bug. Fix it before anything else.

## Branches

`main` stays green. Everything else happens on a branch:

```
feat/<what>      new capability      feat/gnn-model
fix/<what>       something broken    fix/unet-seam-artifact
exp/<what>       throwaway spike     exp/try-fourier-features
```

Small commits, present tense, say why rather than what:

```
fix: pad decoder upsample circularly in theta

Bilinear interpolate clamped at the tensor edge, reintroducing a ~30%
discontinuity across the theta=0 seam that MixedPadConv2d had removed.
```

Open a PR even for small things. It takes ten seconds and it means CI has run
before the code reaches `main`.

## Division of labour

The module boundaries are drawn so that two people can work in parallel without
merge conflicts. Pick a lane and stay in it; if you need something across the
boundary, ask rather than reaching over.

| Lane | Owns | Files |
| --- | --- | --- |
| **A - data & physics** | geometry, the reference solver, dataset generation, features | `geometry/`, `data/` |
| **B - models & serving** | architectures, training loop, metrics, API, demo | `models/`, `training/`, `serving/` |

The demo is lane B's, and it is deliberately a separate module: `serving/demo.py`
plus `serving/static/demo.html` can be deleted without touching the service
contract in `serving/api.py`. Anything the page needs that the service does not
belongs in the demo module, not in `/predict`.

Shared, change by agreement: `config.py`, `cli.py`, `utils.py`.

The contract between the lanes is small and should stay that way:

- Lane A guarantees `extract_features` returns `(n_arc, n_theta, N_FEATURES)`
  float32, and `FEATURE_NAMES` describes it.
- Lane B guarantees every model implements `Surrogate.predict_pa` and returns
  pascals.

If you change `FEATURE_NAMES`, say so - it invalidates every checkpoint, because
the input width changes.

## Before you push

```bash
make fmt     # ruff format + autofix
make lint    # must be clean
make test    # must be green
```

CI runs the same three plus the end-to-end smoke. Nothing merges red.

## What belongs in a test

Not "does this function run". Tests here should pin **properties that would
silently break and produce plausible-looking wrong numbers**:

- rotation invariance of the features
- circular topology of the padding
- train/val/test split disjointness by vessel
- checkpoint round-trip producing identical predictions
- absolute scale tracking `1/r^3` across calibres

Every one of those caught a real bug during the initial build. A test that only
checks a tensor's shape catches nothing.

## Things that are easy to get wrong

**Units.** The API speaks millimetres and millilitres per second; everything
internal is strictly SI. Convert once, at the boundary, in `serving/api.py`.
A factor of 1000 in radius is a factor of a billion in shear.

**Data leakage.** Split by vessel, never by node. `make_splits` already does this
and `test_data.py` enforces it - do not add a "quick" flatten-then-split anywhere.

**Normalisation.** Fit on train only, and persist it with the weights. If you
ever find yourself recomputing a scaler at load time, stop.

**Caching.** Datasets and runs are content-addressed. If you change code in a way
that changes what generation *produces* without changing the config, the cached
dataset is now stale and nothing will tell you. Run `hemoflow data --force`.

**Claims.** The reference solver is reduced-order, not CFD. Never let a number
from this repo be described as clinical accuracy, and never quote batched
throughput as if it were interactive latency.
