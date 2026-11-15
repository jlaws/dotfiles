---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

You are a subagent dispatched by an orchestrator that will paste your final reply verbatim into its context and cannot ask follow-ups. Review only the add function in calc/ops.py for correctness defects (edit nothing). Save your final reply to ./report.md. If you find nothing, do not pad the report with style opinions or hypothetical issues; say so in a single line.
