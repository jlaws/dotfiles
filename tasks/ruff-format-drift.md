# Task: Bring the Python tree back under `ruff format`

## Context
Found by the diff review of the command case audit (2026-11-24). `make check` runs `ruff check` and
`ty`; it does not run `ruff format --check`. `make fix` formats, but nothing fails when a file
drifts.

## Problem
With ruff 0.15.5, `ruff format --diff tests/ evals/ macos_setup/` reports
"34 files would be reformatted, 12 files already formatted" (about 2200 changed lines). None of the
drift comes from one change, so no single PR owns it.

## Scope
1. Run `make fix` on its own branch and review the diff for anything that is not whitespace or
   wrapping.
2. Decide whether `make check` should gain `ruff format --check .` so drift fails the gate.
