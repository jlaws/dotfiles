set -euo pipefail
cp -R "$1/py-calc-planned/." .
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
git add -A
git commit -q -m "Initial commit"
git init -q --bare .remote.git
git remote add origin ./.remote.git
git push -q origin main
printf "%s\n" .remote.git >> .git/info/exclude
cat > tests/test_empty.py <<'PYEOF'
import unittest

from calc.ops import mean


class EmptyMeanTests(unittest.TestCase):
    def test_mean_of_empty_list_is_zero(self):
        self.assertEqual(mean([]), 0.0)


if __name__ == "__main__":
    unittest.main()
PYEOF
cat > calc/ops.py <<'PYEOF'
"""Arithmetic helpers."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def mean(xs):
    """Return the arithmetic mean of xs. The mean of an empty list is 0.0."""
    if not xs:
        return 0.0
    return sum(xs) / len(xs)
PYEOF
cat > plans/plan.md <<'PLANEOF'
# Empty-Mean Fix Implementation Plan

**Purpose:** `mean([])` returns `0.0`, as its docstring promises, and the README says so.

**Codebase Orientation:** `calc/ops.py` holds `add` and `mean`; tests live in `tests/` and run with
`python3 -m unittest`.

## PR Boundaries
- PR 1: Phase 1 (behavior and test).
- PR 2: Phase 2 (documentation). Reviewable on its own once PR 1 has merged.

## Phase 1: Return 0.0 for an empty list

**Goal:** `mean([])` returns `0.0` instead of raising `ZeroDivisionError`.
**Acceptance:** `python3 -m unittest` prints `Ran 3 tests` and `OK`.

### Task 1: Failing test
Add `tests/test_empty.py` asserting `mean([]) == 0.0`; confirm it fails with `ZeroDivisionError`.

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
- [x] Phase 1: Return 0.0 for an empty list
- [ ] Phase 2: Document the behavior

## Decision Log

## Surprises & Discoveries
PLANEOF
printf "notes\n" > scratch.txt
git add -A
git commit -q -m "Return 0.0 for the mean of an empty list"
git push -q origin main
printf 'half-finished thought\n' >> scratch.txt
