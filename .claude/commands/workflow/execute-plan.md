---
name: execute-plan
description: "Execute an implementation plan in batches with review checkpoints. Use when you have a written plan to work through task-by-task."
argument-hint: "<plan file path>"
---

Load and follow the `workflow/executing-plans` skill to execute the implementation plan.

Plan file: $ARGUMENTS

If no arguments provided, look for the most recent plan in `docs/plans/` and confirm with the user before starting.
