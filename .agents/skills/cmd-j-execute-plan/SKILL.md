---
name: cmd-j-execute-plan
description: "Use when invoking the j-execute-plan workflow."
disable-model-invocation: true
---

# Execute Plan

Read the plan file below and review it critically for gaps or contradictions before starting.
Then pick the execution mode without asking: apply `subagent-driven-development`, Execution Mode,
which states the choice and logs it. Load `executing-plans` for inline batches or
`subagent-driven-development` for a fresh subagent per task, then follow that skill exactly
(maintain the plan's living-document ledger, verify each step, stop and ask on blockers).

Do the bare minimum the plan requires, no more: build no feature, capability, or option it did not ask for (`.agents/references/workflow/completeness-principle.md`, Where Thoroughness Stops).

Run without pausing between batches, tasks, or phases. Stop only for a blocking question (`executing-plans`, When to Stop and Ask), to confirm a discovered plan path, or when `$cmd-j-diff-review` has reported the PR URL.

Each PR boundary ends with `$cmd-j-diff-review` (`executing-plans`, Step 5). It fixes everything it finds except large and unrelated items, which it files in the repository's own tracker; then it opens the PR or pushes to the open one, and reports the PR URL and whether another PR remains in the plan. Where the plan spans several PRs, stop after each one and wait for review; `$cmd-j-next` resumes at the next boundary.

Plan file: the user's provided input

If no path is provided, discover regular, non-symlink plan files in `scratchpad/plans/`, then
`${TMPDIR:-/tmp}/j-plan/<repo-id>/` (under `/tmp/j-plan/` when `TMPDIR` is unset). Skip the temp
location unless `j-plan/` and `<repo-id>/` are each a non-symlink directory owned by the current user;
on a shared `/tmp` another user can plant plans there. Show the full paths and modification times, then
ask the user to confirm the chosen path even when there is only one candidate.
**MUST NOT execute a discovered plan without confirmation.** If there are none, ask for a path.
Conversation context is not a plan-file substitute.
