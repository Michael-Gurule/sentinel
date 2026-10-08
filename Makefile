# Developer entry points. Run inside the activated conda env:
#   conda activate sentinel
PYTHON ?= python
ENV_NAME ?= sentinel

.DEFAULT_GOAL := help
.PHONY: help env env-update hooks lint format typecheck test cov check clean

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

cov:  ## Run tests with coverage report
	MPLBACKEND=Agg $(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=xml

check: lint typecheck test  ## Everything CI runs

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage coverage.xml build dist
	find . -type d -name __pycache__ -not -path "./venv/*" -prune -exec rm -rf {} +
