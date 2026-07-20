---
name: j-execute-plan
description: "Execute a written implementation plan task-by-task with verification gates — inline batches or a fresh subagent per task. Use when you have a saved plan ready to implement. Do NOT use for creating the plan (use $cmd-j-plan) or ad-hoc changes without a plan."
argument-hint: "<path to plan file>"
---

Read the plan file below and review it critically for gaps or contradictions before starting. Then pick the execution mode:

- **Inline batches** — load the `executing-plans` skill. Best for small/medium, tightly-coupled plans.
- **Subagent per task** — load the `subagent-driven-development` skill. Best for large plans with mostly-independent tasks, or when inline execution would exhaust context.

Default to inline unless the plan is large or the tasks are clearly independent; state which mode you chose and why, then follow that skill exactly (maintain the plan's living-document ledger, verify each step, stop and ask on blockers).

Plan file: $ARGUMENTS

If no path provided, ask which plan to execute (or check `docs/plans/`).
