---
name: j-execute-plan
description: "Execute a written implementation plan task-by-task with verification gates — inline batches or a fresh subagent per task. Use when you have a saved plan ready to implement. Do NOT use for creating the plan (use $cmd-j-plan) or ad-hoc changes without a plan."
argument-hint: "<path to plan file>"
---

Read the plan file below and review it critically for gaps or contradictions before starting. Then pick the execution mode:

- **Inline batches** — load the `executing-plans` skill. Best for small/medium, tightly-coupled plans.
- **Subagent per task** — load the `subagent-driven-development` skill. Best for large plans with mostly-independent tasks, or when inline execution would exhaust context.

Default to inline unless the plan is large or the tasks are clearly independent; state which mode you chose and why, then follow that skill exactly (maintain the plan's living-document ledger, verify each step, stop and ask on blockers).

Each PR boundary ends by opening the PR — that is `finishing-branch`'s default and needs no prompting. Where the plan spans several PRs, stop after each one and wait for review; `$cmd-j-next` resumes at the next boundary.

Plan file: $ARGUMENTS

If no path is provided, discover regular, non-symlink plan files in `scratchpad/plans/`, then
`${TMPDIR:-/tmp}/j-plan/<repo-id>/` (under `/tmp/j-plan/` when `TMPDIR` is unset). Show the full paths
and modification times, then ask the user to confirm the chosen path even when there is only one
candidate. **MUST NOT execute a discovered plan without confirmation.** If there are none, ask for a
path. Conversation context is not a plan-file substitute.
