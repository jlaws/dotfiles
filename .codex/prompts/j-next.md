---
name: j-next
description: "Advance a multi-PR plan to its next part — confirm the last part landed in origin/main, branch fresh off main, execute the next phases, and open the PR. Use after an earlier PR from the plan was merged. Do NOT use for a plan's first PR (use $cmd-j-execute-plan) or to refresh an already-open PR (use $cmd-j-rebase)."
argument-hint: "[path to plan file]"
---

Advance a multi-PR plan to its next part: confirm the previous part landed on `origin/main`, start a clean branch off the updated trunk, execute the next PR boundary's phases, and open the PR.

Plan file: $ARGUMENTS

If no path is provided, discover regular, non-symlink plan files in `scratchpad/plans/`, then
`${TMPDIR:-/tmp}/j-plan/<repo-id>/` (under `/tmp/j-plan/` when `TMPDIR` is unset). Show the full paths
and modification times, then ask the user to confirm the chosen path even when there is only one
candidate. **MUST NOT execute a discovered plan without confirmation.** If there are none, ask for a
path. Conversation context is not a plan-file substitute.

## Phase 1: Read the plan, find the next part

The plan file is the record of what is done — not this conversation, which predates the merge and may describe a tree that no longer exists.

1. Read the plan file end to end.
2. From its **PR Boundaries** section and its `## Progress` ledger, identify the last completed part and the phases belonging to the next one. If the plan declares no PR boundaries, treat every remaining unchecked phase as this part's scope and say so.
3. If nothing is unchecked, report the plan complete and stop.

## Phase 2: Confirm the previous part landed

Do not ask GitHub — the API is slow, and a squash merge severs ancestry, so `git merge-base --is-ancestor` reports a false negative on a part that did merge. Verify by content.

1. `git status --porcelain` — require a clean tree. If anything is uncommitted or staged, STOP and tell the user to commit or discard first. Do NOT auto-stash.
2. `git fetch origin main`.
3. Get the file set the previous part touched:
   ```bash
   git diff --name-only $(git merge-base HEAD origin/main)..HEAD
   ```
   If HEAD is already a reset `main`, that set is empty — fall back to the files named by the plan's completed phases.
4. Confirm that content is in the trunk:
   ```bash
   git diff origin/main HEAD -- <those files>
   ```
   An empty diff means the previous part is merged.
5. A non-empty diff is not automatically a failure — `origin/main` may carry later edits to the same files. Read it and judge whether the previous part's behavior is present. STOP and show the diff if you cannot tell, or if the part is plainly absent. Building the next part on top of unmerged work produces a PR that silently re-lands or reverts it.

## Phase 3: Clean slate off main

The `writing-plans` between-PRs sequence — run each as a separate command so a failure points at one thing:

```bash
git checkout main
```
```bash
git fetch origin main
```
```bash
git reset --hard origin/main
```
```bash
git branch --merged origin/main | grep -vE '^\*|^\s*(main|master)$' | xargs -r git branch -d
```
```bash
git checkout -b <type>/<short-description> origin/main
```

Name the branch from the next part's phases, `type/short-description`.

## Phase 4: Execute the next part

Load the skills these phases actually need, then execute:

- `executing-plans` (inline batches) or `subagent-driven-development` (fresh subagent per task) — same choice as `$cmd-j-execute-plan`: inline for small or tightly-coupled work, subagents for large or mostly-independent tasks. State which you picked and why.
- `test-driven-development` for each TDD phase.
- `documentation-validation` for the per-phase doc deltas.
- Whatever domain skill the phases name.

Scope stops at this part's PR boundary. Later phases are the next run of this command, not this one.

Maintain the plan file's living-document sections as you go — `## Progress` with commit SHAs, `## Decision Log`, `## Surprises & Discoveries`. They are how the next run of this command finds its place.

Honor the plan's validation gate before moving on, and stop on failure per `executing-plans`.

## Phase 5: Open the PR

Invoke the `create-pr` agent to stage, commit, push, and open the PR for this part. Title and body cover this part's phases only, and the body says which part of the plan it is.

Report the PR URL, the branch, the phases completed, which tests ran with their result, and what remains in the plan.

### Cross-References

- **skill:cmd-j-execute-plan** -- executes a plan from its first part; this command resumes at a later PR boundary
- **skill:cmd-j-rebase** -- refreshes an already-open PR; this command opens the next one
- **skill:writing-plans** -- PR Boundaries, and the between-PRs reset sequence Phase 3 runs
- **skill:executing-plans** -- the living-document ledger Phase 4 maintains
- **agent:create-pr** -- Phase 5 PR creation
