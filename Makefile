.PHONY: help install hooks format lint typecheck test cov validate check clean

PY ?= poetry run

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the project and development dependencies
	poetry install

hooks: ## Install pre-commit hooks
	$(PY) pre-commit install

format: ## Apply Ruff formatting and safe lint fixes
	$(PY) ruff format .
	$(PY) ruff check --fix .

lint: ## Check lint rules and formatting without modifying files
	$(PY) ruff check .
	$(PY) ruff format --check .

typecheck: ## Run mypy in strict mode over the package
	$(PY) mypy src

test: ## Run the test suite
	$(PY) pytest

cov: ## Run the test suite with coverage
	$(PY) pytest --cov=modern_nn_lab --cov-report=term-missing

validate: ## Verify that every registered track has a package, config, and prompt
	$(PY) python scripts/validate_scaffold.py

check: lint typecheck test validate ## Full quality gate, identical to CI

clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
