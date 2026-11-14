# Task: Clean Up Squash-Merged Branches in j-next

## Context
Found while planning `feat/plan-close-review-and-lean-planning`.

## Problem
`git branch --merged origin/main | ... | xargs -r git branch -d` (`.claude/commands/j-next.md:54`, and `writing-plans` Between PRs) never lists squash-merged branches, because a squash merge leaves the branch's commits unreachable from `origin/main`. This repo squash-merges, so the cleanup deletes nothing.

## Scope
1. Detect merged branches through `gh pr view <branch> --json state` or patch-id comparison.
2. Update both copies (j-next and `writing-plans` Between PRs) in every tree.
3. Pin the new detection with a test in `tests/test_agent_config.py`.
