---
type: llm
focus: { source: file, path: plans/plan.md }
---

All claims are required.
PASS if the plan names calc/ops.py as the file to change and a test file under tests/.
PASS if the plan orders a failing test for mean([]) before the code change.
PASS if the plan gives python3 -m unittest as the verification command.
FAIL if the plan proposes changing the docstring instead of the behavior, or omits the test.
