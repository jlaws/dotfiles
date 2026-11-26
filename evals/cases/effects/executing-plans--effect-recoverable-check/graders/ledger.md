---
type: llm
focus: { source: file, path: plans/plan.md }
---

All claims are required.
PASS if the Progress checklist has both tasks ticked and a commit SHA recorded for Phase 1.
PASS if Surprises & Discoveries records each style-check failure the run hit, quoting the "line too long" output and saying how calc/ops.py was changed to fix it; if the check never failed, it says so.
FAIL if the file says the run stopped, is blocked, or is waiting for input because of the style check.
