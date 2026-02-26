# uv Workflows

```bash
# New project
uv init my-project && cd my-project
uv python pin 3.12

# Deps (updates pyproject.toml + uv.lock in one step)
uv add fastapi uvicorn
uv add --dev pytest ruff

# Run without activating venv
uv run pytest
uv run python app.py

# CI/deploy: fail if lockfile stale
uv sync --frozen

# Upgrade
uv lock --upgrade-package requests
uv lock --upgrade  # all deps
```

## uv Key Opinions
- **Always commit `uv.lock`** -- reproducible builds
- **Use `uv run`** instead of activating venvs -- simpler, works in scripts/CI
- **`--frozen` in CI** -- fail if lockfile is stale rather than silently resolving
- **Pin Python version** with `.python-version` file
- **Export for compat**: `uv export --format requirements-txt > requirements.txt`

## uv Gotchas
- `uv add` modifies `pyproject.toml` AND `uv.lock` in one step (unlike poetry two-step)
- `uv sync` creates `.venv` if it doesn't exist
- Cache is global (`~/.cache/uv`) -- shared across projects, rarely needs clearing
- `uv pip install` is pip-compat interface; `uv add` is project-level -- don't mix them

## uv Docker (multi-stage)
```dockerfile
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv
COPY . .
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "app.py"]
```

## uv CI Pattern
```yaml
- uses: astral-sh/setup-uv@v2
  with: { enable-cache: true }
- run: uv python install 3.12
- run: uv sync --all-extras --dev
- run: uv run pytest
```

## uv Workspace (monorepo)
```toml
[tool.uv.workspace]
members = ["packages/*"]
```
