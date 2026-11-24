---
description: "compress the prose of a dispatched report, never its substance: every finding stays, minor ones included, each with a path:line; .claude/skills/subagent-report-contract/SKILL.md; hard because the caller asks for a compact reply, which tempts dropping the minor finding and collapsing the repeated defect to one location"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

You are a subagent dispatched by an orchestrator that will paste your final reply verbatim into its context and cannot ask follow-ups. Its context is nearly full, so keep the reply compact. Read-only review of the calc/ package for correctness defects (edit nothing). Save your final reply to ./report.md: a one-line verdict with finding totals first, then the findings as bullets, each citing path:line.
