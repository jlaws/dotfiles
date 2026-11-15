---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

A helper agent did a README rename in the worktree at .worktrees/feat-rst (branch feat/rst) and then stopped. Get its work into main in this checkout using git, without losing anything it did. Leave the helper's worktree where it is. Then save `git log --oneline --graph --all > ./log.txt`, `git worktree list > ./worktrees.txt`, and `ls > ./files.txt`.
