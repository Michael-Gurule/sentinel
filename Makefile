# Developer entry points. Run inside the activated conda env:
#   conda activate locant
PYTHON ?= python
ENV_NAME ?= locant

.DEFAULT_GOAL := help
.PHONY: help env env-update hooks lint format typecheck test cov bench check demo onnx metrics hero data data-verify data-figures experiments e1 e2 e3 e4 e5 e6 e7 clean

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

env:  ## Create the conda environment from environment.yml
	conda env create -f environment.yml

env-update:  ## Sync the conda environment with environment.yml
	conda env update -n $(ENV_NAME) -f environment.yml --prune

hooks:  ## Install pre-commit git hooks
	$(PYTHON) -m pre_commit install

lint:  ## Lint and check formatting (no changes)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:  ## Auto-fix lint findings and format code
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

typecheck:  ## Static type check (mypy, strict with legacy ratchet)
	$(PYTHON) -m mypy

test:  ## Run the test suite
	MPLBACKEND=Agg $(PYTHON) -m pytest

# Packages whose correctness the project's results depend on; gated at 85%.
CORE_COVERAGE = src/locant/core/*,src/locant/geolocation/*,src/locant/tracking/*,src/locant/fusion/*,src/locant/sim/*,src/locant/data/*,src/locant/detection/*,src/locant/classification/*,src/locant/eval/*,src/locant/pipeline/*,src/locant/runs.py,src/locant/cli.py

cov:  ## Run tests with coverage; enforce >=85% on core packages
	MPLBACKEND=Agg $(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=xml
	$(PYTHON) -m coverage report --include="$(CORE_COVERAGE)" --fail-under=85

bench:  ## Per-stage latency benchmarks with budgets (benchmarks/)
	MPLBACKEND=Agg $(PYTHON) -m pytest benchmarks --benchmark-only --benchmark-columns=min,median,max,rounds

check: lint typecheck cov bench  ## Everything CI runs

demo:  ## Run the full pipeline on the multi-INT scenario (recorded in runs/)
	$(PYTHON) -m locant run configs/scenario/multi_int.yaml --pipeline configs/pipeline/default.yaml

onnx:  ## Export the shipped classifier to ONNX (models/opir_event_classifier/model.onnx)
	$(PYTHON) -m locant export-onnx models/opir_event_classifier

DATASET_CONFIG ?= configs/dataset/opir_v2.yaml
DATASET_DIR ?= data/opir_v2
DATASET_MANIFEST ?= data/manifests/opir_v2.json
WORKERS ?= 4

data:  ## Build the OPIR dataset and update its versioned manifest
	$(PYTHON) -m locant data build --config $(DATASET_CONFIG) --out $(DATASET_DIR) --workers $(WORKERS)
	mkdir -p $(dir $(DATASET_MANIFEST))
	cp $(DATASET_DIR)/manifest.json $(DATASET_MANIFEST)

data-verify:  ## Rebuild the dataset and check it against the versioned manifest
	$(PYTHON) -m locant data verify --config $(DATASET_CONFIG) --out $(DATASET_DIR) --workers $(WORKERS) --manifest $(DATASET_MANIFEST)

hero:  ## Render the README hero figure (docs/figures/hero.png; ~30 s)
	$(PYTHON) -m scripts.hero_figure

data-figures:  ## Render data-card figures from the built dataset
	$(PYTHON) scripts/plot_dataset.py --data $(DATASET_DIR) --out docs/figures

e1:  ## E1 detection experiment (needs `make data`)
	$(PYTHON) -m experiments.e1_detection --workers $(WORKERS)

e2:  ## E2 classification experiment (~1 h on Apple MPS)
	$(PYTHON) -m experiments.e2_classification

e3:  ## E3 uncertainty experiment (needs E1 and E2); exports models/opir_event_classifier
	$(PYTHON) -m experiments.e3_uncertainty --workers $(WORKERS)

e4:  ## E4 geolocation experiment (no dataset needed; ~5 min)
	$(PYTHON) -m experiments.e4_geolocation

e5:  ## E5 tracking experiment (no dataset needed; ~5 min)
	$(PYTHON) -m experiments.e5_tracking

e6:  ## E6 fusion ablation + track classification (needs models/opir_event_classifier; ~8 min)
	$(PYTHON) -m experiments.e6_fusion

e7:  ## E7 robustness: outages, latency, RF bias (no dataset needed; ~8 min)
	$(PYTHON) -m experiments.e7_robustness

metrics:  ## Regenerate reports/metrics.json and every results table in the docs
	$(PYTHON) -m experiments.metrics

experiments: e1 e2 e3 e4 e5 e6 e7 metrics  ## Run all experiments in order, then refresh the docs

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage coverage.xml build dist
	find . -type d -name __pycache__ -not -path "./venv/*" -prune -exec rm -rf {} +
