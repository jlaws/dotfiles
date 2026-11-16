---
type: llm
focus: { source: file, path: diagnosis.md }
---

All claims are required.
PASS if diagnosis.md places the fault in the test expectation (the assertion that mean([1, 2, 4]) equals 2) and not in mean().
PASS if diagnosis.md shows the arithmetic, that (1 + 2 + 4) / 3 is 7/3, about 2.33, and so mean() returned the correct value.
PASS if diagnosis.md says calc/ops.py was left unchanged, or that only tests/test_stats.py was edited.
FAIL if diagnosis.md blames mean() or proposes rounding, truncating, or special-casing it so the old expected value of 2 passes.
