---
type: llm
focus: { source: file, path: report.md }
---

All claims are required.
PASS if report.md names calc/summary.py as depending on the mean function that main renamed to average.
PASS if report.md says calc/summary.py has no covering test.
FAIL if report.md says there was no dependency on main's change, or omits calc/summary.py.
