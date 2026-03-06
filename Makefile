.PHONY: lint format fix typecheck check

lint:
	uv run ruff check .

format:
	uv run ruff format .

fix:
	uv run ruff check --fix . && uv run ruff format .

typecheck:
	uv run mypy setup.py

check: lint typecheck
