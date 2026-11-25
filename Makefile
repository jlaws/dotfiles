VENV := .venv
BIN := $(VENV)/bin
PY_FILES := $(shell find . -name '*.py' -not -path './.venv/*' -not -path './.claude/references/*' -not -path './evals/fixtures/*')

.PHONY: venv lint format format-check fix typecheck check test verify eval-stage eval-lint eval-pilot \
	eval-smoke eval-baseline eval-spend eval-compare eval-headroom eval-affected

venv: $(VENV)/.installed

$(VENV)/.installed: pyproject.toml
	uv sync --group dev
	@touch $@

lint: venv
	$(BIN)/ruff check .

format: venv
	$(BIN)/ruff format .

format-check: venv
	$(BIN)/ruff format --check .

fix: venv
	$(BIN)/ruff check --fix . && $(BIN)/ruff format .

typecheck: venv
	@if [ -n "$(PY_FILES)" ]; then $(BIN)/ty check $(PY_FILES); else echo "No Python files to typecheck"; fi

test: venv
	$(BIN)/python -m unittest discover -s tests -t .

check: format-check lint typecheck

verify: check test

# KB evals (evals/README.md). Stage and lint are free; the targets below them bill an API key.
STAGE_ROOT := $(or $(TMPDIR),/tmp)

eval-stage:
	python3 -m evals.kbeval stage --out "$(STAGE_ROOT)/kbeval-kb" --staging kb
	python3 -m evals.kbeval stage --out "$(STAGE_ROOT)/kbeval-vanilla" --staging vanilla

eval-lint:
	python3 -m evals.kbeval lint

J ?= 8
VANILLA_RUNS ?= 3
VANILLA_MODELS ?= claude-opus-5-5 claude-sonnet-5-5
EXCLUDE_TAGS ?=

eval-pilot:
	KBEVAL_ALLOW_PAID=1 python3 -m evals.kbeval pilot --jobs $(J)

eval-smoke:
	$(if $(ASSETS),,$(error ASSETS is required))
	KBEVAL_ALLOW_PAID=1 python3 -m evals.kbeval smoke --assets $(ASSETS) --jobs $(J)

eval-baseline:
	$(if $(MODEL),,$(error MODEL is required))
	KBEVAL_ALLOW_PAID=1 python3 -m evals.kbeval baseline --model $(MODEL) --jobs $(J) \
		--vanilla-runs $(VANILLA_RUNS) --vanilla-models $(VANILLA_MODELS) \
		$(if $(CAP),--cap-usd '$(CAP)') \
		$(foreach tag,$(EXCLUDE_TAGS),--exclude-tag $(tag))

eval-spend:
	python3 -m evals.kbeval spend

eval-compare:
	$(if $(OLD),,$(error OLD is required))
	$(if $(NEW),,$(error NEW is required))
	python3 -m evals.kbeval compare $(OLD) $(NEW)

eval-headroom:
	@python3 -m evals.kbeval headroom

eval-affected:
	@python3 -m evals.kbeval affected
