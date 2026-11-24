---
description: "a user wants a bloated, tested service and a deeply nested method tidied without changing behavior; .claude/skills/refactoring-and-debt/SKILL.md; should"
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

OrderService has 40 methods and calculateShipping has a twelve-level nested if. Tests exist and pass. Tidy it up without changing behavior.
