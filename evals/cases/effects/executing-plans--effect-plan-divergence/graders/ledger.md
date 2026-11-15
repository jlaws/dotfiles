---
type: llm
focus: { source: file, path: plans/plan.md }
---

All claims are required.
PASS if the file has a Progress checklist with all three tasks ticked and a commit SHA recorded for each of the two phases.
PASS if Surprises & Discoveries (or the Decision Log) records that the suite ran 3 tests where the plan expected 4, quoting the actual output.
PASS if the file still contains the original three tasks and their steps, not a rewritten plan.
FAIL if the mismatch between 3 tests and the plan's 4 is not recorded anywhere in the file.
