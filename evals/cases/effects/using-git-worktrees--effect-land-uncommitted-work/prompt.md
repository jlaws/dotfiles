---
description: "commit a worktree's uncommitted changes before integrating with a merge, and leave the worktree in place; .claude/skills/using-git-worktrees/SKILL.md; hard because the helper's rename is only staged, so merging the branch as it stands lands nothing"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

A helper agent did a README rename in the worktree at .worktrees/feat-rst (branch feat/rst) and then stopped. Get its work into main in this checkout with a git merge commit, without losing anything it did. Leave the helper's worktree where it is. Then save `git log --oneline --graph --all > ./log.txt`, `{ git rev-parse feat/rst; git log --merges --format=%P main; } > ./merges.txt`, `git worktree list > ./worktrees.txt`, and `ls > ./files.txt`.
