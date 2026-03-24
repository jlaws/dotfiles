# uv Workflows

Fast Python package and environment manager. Replaces pip, poetry, and pipenv in most workflows.

## Project Setup

```bash
# New project
uv init my-project
cd my-project
uv python pin 3.12

# Add dependencies (updates pyproject.toml + uv.lock in one step)
uv add fastapi uvicorn
uv add --dev pytest ruff mypy

# Run without activating venv (preferred)
uv run pytest
uv run python app.py

# Or activate venv for interactive work
source .venv/bin/activate
python app.py
```

## Dependency Management

```bash
# Add specific version
uv add 'requests>=2.28.0'

# Add optional dependencies (for different use cases)
uv add --optional data pandas polars
# pyproject.toml: [project.optional-dependencies]
uv sync --all-extras  # install all optional deps

# Upgrade packages
uv lock --upgrade-package requests  # one package
uv lock --upgrade  # all dependencies

# Check for outdated packages
uv pip list --outdated

# Remove dependency
uv remove requests
```

## CI/CD Deployment

```bash
# MUST use --frozen in CI (fail if lockfile stale)
uv sync --frozen
uv sync --frozen --no-dev  # production: skip dev deps
uv sync --frozen --all-extras  # include optional deps
```

**Critical**: Always commit `uv.lock`; regenerate only on dependency changes.

## Tool Management (uv >= 0.2)

```bash
# Install CLI tools globally
uv tool install ruff
uv tool install black
uv tool install pytest

# Run tool without global install (like npx)
uv tool run ruff check .
uv tool run pytest --version

# Update tool
uv tool upgrade ruff

# List installed tools
uv tool list
```

## Script Dependencies (PEP 723)

```python
# standalone.py with embedded metadata (Python 3.11+)
# /// script
# dependencies = ["requests>=2.28.0", "click>=8.0"]
# ///

import requests
import click

@click.command()
@click.argument('url')
def fetch(url):
    resp = requests.get(url)
    click.echo(resp.text)

if __name__ == '__main__':
    fetch()
```

```bash
# Run script with auto-installed dependencies
uv run standalone.py https://example.com
```

## Python Version Management

```bash
# List available Python versions
uv python list

# Install specific version
uv python install 3.12
uv python install 3.11

# Pin in project (.python-version file)
uv python pin 3.12

# Use in CI
uv python install 3.12 3.11
uv run --python 3.11 pytest  # run with specific version
```

## Migration from pip/poetry/pipenv

```bash
# From pip/requirements.txt
uv sync  # reads requirements.txt if no pyproject.toml

# From poetry/Pipenv
uv import poetry/Pipenv  # convert lock file to uv.lock

# Interop: export to requirements.txt
uv export --format requirements-txt > requirements.txt
uv export --format requirements-txt --no-dev > requirements-prod.txt
```

## Docker Deployment

```dockerfile
# Multi-stage: small final image
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

## GitHub Actions CI Pattern

```yaml
- uses: astral-sh/setup-uv@v2
  with:
    enable-cache: true
    python-version: "3.12"
    cache-dependency-glob: "**/uv.lock"

- run: uv sync --frozen --all-extras --dev
- run: uv run pytest
- run: uv run ruff check .
- run: uv run mypy .
```

## Workspaces (Monorepo)

```toml
# pyproject.toml (root)
[tool.uv.workspace]
members = ["packages/api", "packages/cli", "packages/shared"]
```

```bash
# Install all workspace packages
uv sync

# Upgrade specific workspace member
uv add -p api requests@^2.30.0
```

## Key Opinions

- **Always commit `uv.lock`** -- reproducible builds, deterministic deploys
- **Use `uv run`** instead of activating venvs -- works in CI, Docker, scripts
- **`--frozen` in CI** -- fail if lockfile is stale; catch accidental changes
- **Pin Python version** with `uv python pin 3.12`
- **Export for compatibility**: `uv export > requirements.txt` for legacy tooling

## Gotchas & Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `uv add` requires lockfile regeneration | One-step (unlike poetry) | Normal; commit both files |
| `.venv` created unexpectedly | `uv sync` creates if missing | Use `uv run` to avoid |
| Cache bloat (`~/.cache/uv`) | Shared across projects | Rarely needs cleanup; safe to delete |
| `uv pip install` vs `uv add` confusion | Different scopes (global vs local) | Use `uv add` for projects, avoid mixing |
| Python version not found | uv doesn't auto-download | Run `uv python install 3.12` first |
| Lock file conflicts in git | Team member used different deps | Regenerate: delete `uv.lock`, run `uv sync` |
