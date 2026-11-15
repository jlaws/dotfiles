---
type: llm
focus: { source: file, path: upgrade-plan.md }
---

All claims are required.
PASS if every dependency with more than one major version to cross (express 3 to 5, mongoose 5 to 8, jest 26 to 29, Node 12 to 22) is split into one step per major version, with no single step jumping two majors.
PASS if the runtime comes first and the framework and database layer come before the test tooling.
PASS if each step has its own branch and a tag or equivalent marker on the pre-step state for rollback.
PASS if the plan records a baseline test run (pass count and coverage) before the first step and says to read the migration notes for each intermediate major version, not only the target.
FAIL if the plan upgrades several of these dependencies in the same step.
