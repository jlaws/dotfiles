---
name: cmd-j-execute-plan
description: "Use when invoking the j-execute-plan workflow."
disable-model-invocation: true
---

# Execute Plan

Read the plan file below and review it critically for gaps or contradictions before starting. Then pick the execution mode without asking: apply `subagent-driven-development`, Mode Selection, to the
plan and state the mode plus the signals that decided it in one line. Load `executing-plans` for inline
batches or `subagent-driven-development` for a fresh subagent per task, then follow that skill exactly
(maintain the plan's living-document ledger, verify each step, stop and ask on blockers).

Run without pausing between batches, tasks, or phases. Stop only for a blocking question (`executing-plans`, When to Stop and Ask) or when the PR is open for review.

Each PR boundary ends by opening the PR — that is `finishing-branch`'s default and needs no prompting. Where the plan spans several PRs, stop after each one and wait for review; `$cmd-j-next` resumes at the next boundary.

Plan file: the user's provided input

If no path is provided, discover regular, non-symlink plan files in `scratchpad/plans/`, then
`${TMPDIR:-/tmp}/j-plan/<repo-id>/` (under `/tmp/j-plan/` when `TMPDIR` is unset). Show the full paths
and modification times, then ask the user to confirm the chosen path even when there is only one
candidate. **MUST NOT execute a discovered plan without confirmation.** If there are none, ask for a
path. Conversation context is not a plan-file substitute.
