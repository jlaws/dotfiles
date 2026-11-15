---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

I'm midway through some edits in this checkout: leave them exactly as they are, staged and uncommitted. Start the median work for calc in a separate working tree inside this repo directory, on a new branch feat/median, so I can keep working here. Make sure nothing about the worktree can end up committed by accident. Write ./report.md with the branch, the worktree path, and the commit it was branched from and why you chose that commit. Finish with `git worktree list > ./worktrees.txt` and `git status --porcelain > ./status.txt`.
