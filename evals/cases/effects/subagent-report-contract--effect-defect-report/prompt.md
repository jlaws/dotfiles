---
description: "a dispatched report leads with a verdict line, cites path:line per finding including minor gaps, quotes error output byte-for-byte, and does not narrate the search; .claude/skills/subagent-report-contract/SKILL.md; hard because a model paraphrases the traceback and drops the minor test gap"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

You are a subagent dispatched by an orchestrator that will paste your final reply verbatim into its context and cannot ask follow-ups. Read-only audit of this repository (edit nothing): find every defect or gap in calc/ and tests/, including minor ones. Run the code to reproduce any failure. Save your final reply to ./report.md. The first line is a one-line verdict with finding totals. Each finding is one bullet citing path:line. Quote any error output exactly as printed. Add a short line for any alternative you considered and rejected. Do not narrate your search.
