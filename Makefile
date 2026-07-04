VENV := .venv
BIN := $(VENV)/bin
PY_FILES := $(shell find . -name '*.py' -not -path './.venv/*' -not -path './.claude/references/*')

.PHONY: venv lint format fix typecheck check test verify

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
	@if [ -n "$(PY_FILES)" ]; then $(BIN)/ty check $(PY_FILES); else echo "No Python files to typecheck"; fi

test: venv
	$(BIN)/python -m unittest discover -s tests -t .

check: lint typecheck

verify: check test
