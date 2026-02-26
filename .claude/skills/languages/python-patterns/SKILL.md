---
name: python-patterns
description: "Python project setup, concurrency model selection, async patterns, and dependency management with uv. Use when creating Python projects, choosing concurrency models, async/await patterns, optimizing performance, or managing dependencies with uv. Do NOT use for Pydantic model design (use pydantic-and-data-validation) or FastAPI application architecture (use fastapi-templates)."
---

# Python Patterns

Tooling opinions, concurrency decisions, async guardrails, profiling workflow, and packaging.

## Style Guide

Source: Google Python Style Guide. Only rules linters/formatters cannot enforce.

### Naming
- `_internal` prefix for module-internal names
- Avoid single-char names except counters (`i`, `j`), exceptions (`e`), file handles (`f`)
- Names describe purpose, not type: `user_list` not `l`, `count` not `n`
- Boolean variables/functions: `is_`, `has_`, `can_`, `should_` prefix
- Avoid generic names: `data`, `info`, `temp`, `val` — be specific
- Module names: short, lowercase, no dashes — `utilities` not `my-utils`

### Docstrings
- Google-style with `Args:`/`Returns:`/`Raises:` sections
- Every public module, function, class, and method

### Practices
- No mutable default args; use `None` + assign inside
- No `assert` for validation — use `raise ValueError`
- Logging: `%` formatting not f-strings (`logger.info('Val: %s', val)`)
- Comprehensions: simple only, no multiple `for` clauses
- Lambda: one-liners only, prefer named functions
- `with` for all file/resource handling
- Max function length ~40 lines
- No `staticmethod`; limit `classmethod` to named constructors
- Properties: only trivial computations

## Tooling Defaults

| Concern | Use | Why |
|---------|-----|-----|
| Package manager | `uv` | 10-100x faster than pip/poetry, handles venvs + Python versions |
| Linter + formatter | `ruff` | Replaces black, isort, flake8 in one tool |
| Type checker | `mypy` (strict) | Catch bugs at dev time |
| Testing | `pytest` + `pytest-asyncio` | De facto standard |
| Build backend | `hatchling` (libraries), `setuptools` (apps) | Hatch is modern, setuptools is universal |

### ruff config opinions
```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

## uv Workflows

See references/uv-workflows.md for uv commands, Docker multi-stage builds, CI integration, and workspace setup.

## Project Scaffolding

- **Always use `src/` layout** -- prevents importing uninstalled code, cleaner test isolation
- **Single source of truth**: `pyproject.toml` for everything (no setup.py, setup.cfg)
- **Version**: `setuptools-scm` for git-tag-based, or `__version__` in `__init__.py`
- **Dependency ranges**: `"requests>=2.28,<3"` -- avoid exact pins except in lockfiles
- **Type stubs**: include `py.typed` marker for PEP 561

### Project Type Selection

| Type | When | Key deps |
|------|------|----------|
| **FastAPI** | REST APIs, microservices, async | fastapi, uvicorn, pydantic-settings, sqlalchemy, alembic |
| **Django** | Full-stack web, admin, ORM-heavy | django, django-environ, psycopg, gunicorn |
| **Library** | Reusable packages | hatchling (build backend) |
| **CLI** | Command-line tools | typer, rich |

## Concurrency Decision Framework

| Workload | Use | Why |
|----------|-----|-----|
| I/O-bound (HTTP, DB, files) | `asyncio` | Single-threaded, no GIL contention, lowest overhead |
| I/O-bound + sync libraries | `threading` + `ThreadPoolExecutor` | When you can't go async all the way |
| CPU-bound | `multiprocessing` | Bypasses GIL, true parallelism |
| CPU-bound + shared state | `multiprocessing` + `Manager` | Avoid; redesign to message-passing if possible |
| Mixed I/O + CPU | `asyncio` + `run_in_executor` | Async for I/O, thread/process pool for CPU |

## Async Patterns

See references/async-patterns.md for async decision table, gather/TaskGroup, semaphore, timeouts, and cheat sheet.

## Profiling & Performance

See references/performance.md for profiling tools, tracemalloc, caching decisions, __slots__, and batch I/O patterns.

## Packaging

- **Build backend**: `hatchling` for libraries, `setuptools` for apps that don't publish
- **Entry points**: `[project.scripts]` for CLIs, `[project.entry-points]` for plugins

### Publishing workflow
```bash
uv pip install build twine
python -m build
twine check dist/*
twine upload --repository testpypi dist/*  # test first
twine upload dist/*
```

## References

- .claude/references/ai-ml/jax-patterns.md — JAX-specific Python patterns
- .claude/references/data/spark-optimization.md — PySpark optimization patterns
