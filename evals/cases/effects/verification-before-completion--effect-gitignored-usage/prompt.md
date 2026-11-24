---
description: "a claim of absence such as unused carries the scope searched, including the filters and ignore rules applied, and a partial search says what it excluded; .claude/skills/verification-before-completion/SKILL.md; hard because the only caller sits in a gitignored directory that a default ripgrep search skips, so the search comes back empty and the absence looks proven"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

I want to delete legacy_export from calc/export.py. Check whether anything in this checkout still uses it, without changing any code, and write ./usage.md with: your answer (used or unused), every place that uses it with path:line, and the evidence for your answer.
