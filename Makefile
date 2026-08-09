.PHONY: install test lint typecheck check

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check .
	poetry run ruff format --check .

typecheck:
	poetry run mypy src

check: lint typecheck test
