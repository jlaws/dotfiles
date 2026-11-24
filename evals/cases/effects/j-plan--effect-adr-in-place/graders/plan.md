---
type: llm
focus: { source: file, path: plans/plan.md }
---

All claims are required.
PASS if the plan has a task that edits docs/adr/calc/empty-input.md in place so that its Decision says mean([]) raises ValueError.
PASS if that task records returning 0.0 as ruled out in the ADR's Ruled Out table and bumps the ADR's updated date.
FAIL if the plan creates a new ADR file for this decision instead of editing docs/adr/calc/empty-input.md, or deletes that file.
FAIL if the plan has no failing test for mean([]) raising ValueError ordered before the change to calc/ops.py.
