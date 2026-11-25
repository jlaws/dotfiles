# Task: Settle the documented Python floor

## Context
Found while planning the tasks/ backlog (2026-11-25). Two docs and the code disagree on the Python
version `setup.sh` needs.

## Problem
Root `CLAUDE.md` (Setup & Installation section) and `README.md` (the `setup.sh` shim paragraph)
say "Python 3.12+". `pyproject.toml` sets ruff `target-version = "py39"`, and
`docs/adr/workflow/merge-agent-configs-during-setup.md` holds the sync code (and the eval harness)
to bare macOS system Python 3.9.6, stdlib only.

## Scope
1. Decide which floor is true for `macos_setup` and `evals/kbeval`.
2. Make root `CLAUDE.md`, `README.md`, `pyproject.toml`, and the ADR agree.
