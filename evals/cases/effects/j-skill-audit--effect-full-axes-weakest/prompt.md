---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-skill-audit audit everything in this repo's .claude/ tree and save the report to ./audit-full.md. Lead with the failure and warning counts from the mechanical check, with every finding by rule id and file path. Then rate each of groundedness, coverage, freshness, structure, and evidence with a one-line reason, name the weakest of the five, and close with the single highest-value next change. Do not read transcripts or auto-memory, and do not edit any file.
