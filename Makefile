VENV := .venv
BIN := $(VENV)/bin
PY_FILES := $(shell find . -name '*.py' -not -path './.venv/*' -not -path './.claude/references/*')

.PHONY: venv lint format fix typecheck check

venv: $(VENV)/.installed

$(VENV)/.installed: pyproject.toml
	uv sync --group dev
	@touch $@

lint: venv
	$(BIN)/ruff check .

format: venv
	$(BIN)/ruff format .

fix: venv
	$(BIN)/ruff check --fix . && $(BIN)/ruff format .

typecheck: venv
	@if [ -n "$(PY_FILES)" ]; then $(BIN)/mypy $(PY_FILES); else echo "No Python files to typecheck"; fi

check: lint typecheck
