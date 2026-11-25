---
type: llm
focus: { source: file, path: plans/plan.md }
---

All claims are required.
PASS if the Progress checklist has both tasks ticked and a commit SHA recorded for Phase 1.
PASS if Surprises & Discoveries records that the style check failed on the plan's docstring line, quoting the "line too long" output, and says how it was fixed.
FAIL if the file says the run stopped, is blocked, or is waiting for input because of the style check.
