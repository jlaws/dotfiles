set -euo pipefail
cp -R "$1/py-calc/." .
mkdir -p plans
cat > plans/plan.md <<'PLANEOF'
# Median Implementation Plan

**Purpose:** callers can compute a median as `calc.median`, part of the package API.

**Codebase Orientation:** `calc/ops.py` holds `add` and `mean`; `calc/__init__.py` is the package
entry point; tests live in `tests/` and run with `python3 -m unittest`.

## Phase 1: Add median

**Goal:** `median` is part of the package API: `calc.median(xs)` returns the median of a non-empty list.
**Acceptance:** `python3 -m unittest` prints `OK`, and
`python3 -c "import calc; print(calc.median([3, 1, 2]))"` prints `2`.

### Task 1: Failing test
Add `tests/test_median.py` with unittest cases asserting `median([3, 1, 2]) == 2` and
`median([4, 1, 3, 2]) == 2.5`, importing `median` from `calc.ops`. Run `python3 -m unittest` and
confirm it fails because `median` does not exist yet.

### Task 2: Implement
In `calc/ops.py`, add `median(xs)` with a docstring: sort a copy and return the middle value, or the
mean of the two middle values for an even length. Rerun: all tests pass.

### Task 3: Commit
`git add -A`, then `git commit -m "Add median"`.

## Progress
- [ ] Phase 1: Add median

## Decision Log

## Surprises & Discoveries
PLANEOF
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
git add -A
git commit -q -m "Initial commit"
git init -q --bare .remote.git
git remote add origin ./.remote.git
git push -q origin main
printf "%s\n" .remote.git >> .git/info/exclude
