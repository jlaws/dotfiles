# Empty-Mean Fix Implementation Plan

**Purpose:** `mean([])` returns `0.0`, as its docstring promises, and the README says so.

**Codebase Orientation:** `calc/ops.py` holds `add` and `mean`; tests live in `tests/` and run with
`python3 -m unittest`.

## Phase 1: Return 0.0 for an empty list

**Goal:** `mean([])` returns `0.0` instead of raising `ZeroDivisionError`.
**Acceptance:** `python3 -m unittest` prints `Ran 3 tests` and `OK`.

### Task 1: Failing test
Add `tests/test_empty.py` with a unittest case asserting `mean([]) == 0.0`. Run
`python3 -m unittest` and confirm it fails with `ZeroDivisionError`.

### Task 2: Fix
In `calc/ops.py`, return `0.0` when `xs` is empty. Rerun: all tests pass.

### Task 3: Commit
`git commit -am "Return 0.0 for the mean of an empty list"` (add the new test file first).

## Phase 2: Document the behavior

**Goal:** README states the empty-list behavior.
**Acceptance:** `README.md` contains the sentence "The mean of an empty list is 0.0."

### Task 1: Edit README
Add the sentence "The mean of an empty list is 0.0." under the usage text.

### Task 2: Commit
`git commit -am "Document the empty-list mean"`

## Progress
- [ ] Phase 1: Return 0.0 for an empty list
- [ ] Phase 2: Document the behavior

## Decision Log

## Surprises & Discoveries
