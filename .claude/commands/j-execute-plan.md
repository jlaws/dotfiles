---
name: j-execute-plan
description: "Execute a written implementation plan task-by-task with verification gates — inline batches or a fresh subagent per task, picked from the plan. Use when you have a saved plan ready to implement. Do NOT use for creating the plan (use /j-plan) or ad-hoc changes without a plan."
argument-hint: "<path to plan file>"
---

Read the plan file below and review it critically for gaps or contradictions before starting.
Then pick the execution mode without asking: apply `subagent-driven-development`, Execution Mode,
which states the choice and logs it. Load `executing-plans` for inline batches or
`subagent-driven-development` for a fresh subagent per task, then follow that skill exactly
(maintain the plan's living-document ledger, verify each step, stop and ask on blockers).

Do the bare minimum the plan requires, no more: build no feature, capability, or option it did not ask for (`.claude/references/workflow/completeness-principle.md`, Where Thoroughness Stops).

Run without pausing between batches, tasks, or phases. Stop only for a blocking question (`executing-plans`, When to Stop and Ask) or when /j-diff-review has reported the PR URL.

Each PR boundary ends with /j-diff-review (`executing-plans`, Step 5). It fixes everything it finds except large and unrelated items, which it files in the repository's own tracker; then it opens the PR or pushes to the open one, and reports the PR URL and whether another PR remains in the plan. Where the plan spans several PRs, stop after each one and wait for review; /j-next resumes at the next boundary.

Plan file: $ARGUMENTS

If no path provided, use the plan file that plan mode assigned this session. Outside plan mode, check `~/.claude/plans/` for entries prefixed with this repo's name — that directory is shared across every project, so a plan from another repo is not a candidate. Ask if more than one fits.
