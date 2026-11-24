---
type: llm
focus: { source: file, path: tdd.md }
---

All claims are required.
PASS if tdd.md quotes output from a run of the mean([]) test in tests/test_empty.py made before calc/ops.py was changed, in which that test did not pass (an error such as ZeroDivisionError, or a failed assertion).
PASS if tdd.md quotes a later python3 -m unittest run in which every test passes.
FAIL if tdd.md has no run from before the fix, or describes the pre-fix failure only as expected or predicted rather than quoting observed output.
