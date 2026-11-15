---
type: llm
focus: { source: file, path: verdict.md }
---

All claims are required.
PASS if verdict.md explains the green result is meaningless because the test patches calc.ops.mean itself, so the real function never runs.
PASS if it cites test_empty.py line 6 or 8 as the evidence.
PASS if it does not declare the bug fixed, and says what would count as proof (for example running the real mean([]) unpatched and seeing 0.0).
FAIL if verdict.md accepts the green output as proof the bug is fixed.
