---
description: "propose 2-3 approaches with a recommendation, state numbered falsifiable assumptions, and raise a conflict the codebase holds on a point the design depends on; .claude/skills/design-first/SKILL.md; hard because the mean() docstring and body disagree on the empty-list case, which only a model that explores the code first will see"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

I want calc to support percentiles, for example percentile(xs, 90). I have not decided anything yet and nothing should be implemented in this session. Work out the design and save it to ./design.md. It must contain: two or three approaches with trade-offs and a clear recommendation; a numbered list of the assumptions you are making, each one stated so it could be proven wrong; and how the design handles an empty list. Do not create or edit any file other than ./design.md. If the existing code or docs conflict with each other on a point your design depends on, say so in the design.
