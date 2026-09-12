"""Reproducibility and bookkeeping helpers shared by every entry point."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every RNG we touch.

    Determinism is a precondition for the ablation table in the README: if two
    runs of the same config disagree, no comparison between configs means
    anything.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def to_plain(obj: Any) -> Any:
    """Recursively convert dataclasses / paths / numpy scalars to JSON-safe types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_plain(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def fingerprint(obj: Any, length: int = 10) -> str:
    """Stable short hash of any config-like object.

    Used as the run id, so an artifact directory name tells you exactly which
    config produced it and re-running the same config resumes rather than
    silently forking a second copy.
    """
    payload = json.dumps(to_plain(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def git_sha(default: str = "unknown") -> str:
    """Current commit, recorded in every manifest so results trace to code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return default
    return out.stdout.strip() or default


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain(payload), indent=2, sort_keys=True) + "\n")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())
