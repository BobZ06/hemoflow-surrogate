#!/usr/bin/env bash
# One-command environment setup.
#
# Installs the CPU build of torch from the PyTorch wheel index (~200 MB instead
# of the ~2.5 GB CUDA build on PyPI). If you have an NVIDIA GPU, drop the
# --index-url and install torch normally first.
set -euo pipefail

python -m pip install --upgrade pip

if ! python -c "import torch" 2>/dev/null; then
  echo "installing torch (CPU build)..."
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

python -m pip install -e ".[all]"

echo
python - <<'PY'
import importlib
required = ["numpy", "scipy", "yaml", "torch", "fastapi", "pytest"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
print("missing:", missing if missing else "none")
PY
echo
echo "setup complete. Try: make smoke"
