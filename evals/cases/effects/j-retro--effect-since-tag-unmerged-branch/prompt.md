---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-retro write a retrospective for everything on main since the v1.0.0 tag. Save it to ./retro-since-v1.md. It must state the number of merge commits, list each reverted commit and what it reverted, list every local branch whose commits are not merged into main with how many commits it is ahead, and give a top-files table with the number of commits that touched each file. Word churn findings as possible instability, not proven problems, and keep it about one engineer (no per-person breakdown).
