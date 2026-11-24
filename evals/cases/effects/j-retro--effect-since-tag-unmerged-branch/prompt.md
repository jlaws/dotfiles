---
description: "for a since-tag window, report merges, reverts, long-lived unmerged branches, and top-churn files from git, with churn worded as potential instability; .claude/commands/j-retro.md; hard because the unmerged branch is invisible in a main-only log, and its commits would inflate the file counts if read from all branches"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-retro write a retrospective for everything on main since the v1.0.0 tag. Save it to ./retro-since-v1.md. It must state the number of merge commits, list each reverted commit and what it reverted, list every local branch whose commits are not merged into main with how many commits it is ahead, and give a top-files table with the number of commits that touched each file. Word churn findings as possible instability, not proven problems, and keep it about one engineer (no per-person breakdown).
