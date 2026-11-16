---
type: llm
focus: { source: file, path: debug-report.md }
---

All claims are required.
PASS if the report names test_mean_of_empty_list_is_zero as the failing test and quotes a ZeroDivisionError.
PASS if the root cause is stated as mean dividing by len(xs) when the list is empty.
PASS if the report cites evidence the fix works, such as a passing test run after the change.
FAIL if the report changes the test's expectation instead of fixing calc/ops.py.
