# Developer entry points. Run inside the activated conda env:
#   conda activate sentinel
PYTHON ?= python
ENV_NAME ?= sentinel

.DEFAULT_GOAL := help
.PHONY: help env env-update hooks lint format typecheck test cov check data data-verify data-figures clean

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
CORE_COVERAGE = src/sentinel/core/*,src/sentinel/geolocation/*,src/sentinel/tracking/*,src/sentinel/fusion/*,src/sentinel/sim/*,src/sentinel/data/*

cov:  ## Run tests with coverage; enforce >=85% on core packages
	MPLBACKEND=Agg $(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=xml
	$(PYTHON) -m coverage report --include="$(CORE_COVERAGE)" --fail-under=85

check: lint typecheck cov  ## Everything CI runs

DATASET_CONFIG ?= configs/dataset/opir_v2.yaml
DATASET_DIR ?= data/opir_v2
DATASET_MANIFEST ?= data/manifests/opir_v2.json
WORKERS ?= 4

data:  ## Build the OPIR dataset and update its versioned manifest
	$(PYTHON) -m sentinel.data --config $(DATASET_CONFIG) --out $(DATASET_DIR) --workers $(WORKERS)
	mkdir -p $(dir $(DATASET_MANIFEST))
	cp $(DATASET_DIR)/manifest.json $(DATASET_MANIFEST)

data-verify:  ## Rebuild the dataset and check it against the versioned manifest
	$(PYTHON) -m sentinel.data --config $(DATASET_CONFIG) --out $(DATASET_DIR) --workers $(WORKERS) --verify $(DATASET_MANIFEST)

data-figures:  ## Render data-card figures from the built dataset
	$(PYTHON) scripts/plot_dataset.py --data $(DATASET_DIR) --out docs/figures

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage coverage.xml build dist
	find . -type d -name __pycache__ -not -path "./venv/*" -prune -exec rm -rf {} +
