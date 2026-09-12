.PHONY: help install lint fmt test data train eval bench serve demo ablation smoke clean

PY ?= python
CONFIG ?= configs/mlp.yaml

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with all extras in editable mode
	$(PY) -m pip install -e ".[all]"

lint:  ## Static checks
	ruff check src tests
	ruff format --check src tests

fmt:  ## Autoformat
	ruff format src tests
	ruff check --fix src tests

test:  ## Run the test suite
	pytest

data:  ## Generate the synthetic training dataset
	$(PY) -m hemoflow.cli data --config $(CONFIG)

train:  ## Train the surrogate defined by CONFIG
	$(PY) -m hemoflow.cli train --config $(CONFIG)

eval:  ## Evaluate the latest checkpoint on the held-out test split
	$(PY) -m hemoflow.cli eval --config $(CONFIG)

bench:  ## Benchmark inference latency against the CFD reference cost
	$(PY) -m hemoflow.cli bench --config $(CONFIG)

ablation:  ## Train the no-physics-prior checkpoint the demo switch needs
	$(PY) -m hemoflow.cli train --config configs/mlp_no_physics.yaml

serve:  ## Launch the inference API on :8000
	uvicorn hemoflow.serving.api:app --host 0.0.0.0 --port 8000

demo:  ## Serve the interactive demo at http://127.0.0.1:8000/
	@echo "demo at http://127.0.0.1:8000/  (ctrl-c to stop)"
	$(PY) -m uvicorn hemoflow.serving.api:app --host 127.0.0.1 --port 8000

smoke:  ## Tiny end-to-end run: data -> train -> eval -> bench
	bash scripts/smoke.sh

clean:
	rm -rf artifacts runs .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
