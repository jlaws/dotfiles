---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-retro write a retrospective for the main branch covering 2026-11-01 through 2026-11-14, compared with the previous 14 days (2026-10-18 through 2026-10-31). Save it to ./retro.md. Include a metrics table with commits, files touched, lines added, and lines removed for both periods plus a trend column, a What Shipped section grouped by theme, a high-churn files table with change counts, and an observations section. Count only commits reachable from main, take every number from git rather than estimating, and keep it about one engineer (no per-person breakdown).
