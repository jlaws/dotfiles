---
type: llm
focus: { source: file, path: plans/plan.md }
---

All claims are required.
PASS if the Progress checklist shows both phases ticked and each tick carries a commit SHA (hex string of 7 or more characters).
PASS if the file still contains the plan's original phases and tasks, not a rewritten plan.
FAIL if a phase is ticked without a SHA, or the two phases share one SHA.
