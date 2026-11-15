---
type: llm
focus: { source: file, path: HANDOFF.md }
---

All claims are required.
PASS if HANDOFF.md records both rejected approaches (ValueError and None) with the reason each was dropped.
PASS if HANDOFF.md gives the exact command python3 -m unittest (or python3 -m unittest tests.test_empty) and the exact failure text ZeroDivisionError: division by zero.
PASS if HANDOFF.md lists files touched with a status for each, saying calc/ops.py is not yet edited and tests/test_empty.py is added.
PASS if HANDOFF.md has numbered next steps in the stated order and lists the README question as an open issue.
PASS if HANDOFF.md is under 100 lines and contains the branch name fix/empty-mean.
FAIL if the file claims calc/ops.py is already fixed.
