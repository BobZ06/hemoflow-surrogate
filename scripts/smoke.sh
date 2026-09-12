#!/usr/bin/env bash
# End-to-end smoke test on a tiny config: data -> train -> eval -> bench.
#
# Unit tests can all pass while the pipeline is broken where the modules meet,
# so this runs the actual CLI the way a person would. It should finish in well
# under a minute on a laptop CPU.
set -euo pipefail

CONFIG="${1:-configs/smoke.yaml}"

echo "==> config"
hemoflow info --config "$CONFIG" | tail -4

echo
echo "==> generate dataset"
hemoflow data --config "$CONFIG"

echo
echo "==> train"
hemoflow train --config "$CONFIG"

echo
echo "==> benchmark against baselines"
hemoflow eval --config "$CONFIG" --all-runs

echo
echo "==> inference latency"
hemoflow bench --config "$CONFIG"

echo
echo "smoke passed"
