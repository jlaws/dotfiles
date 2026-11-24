---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "a user designing a pytest suite with fixtures and mocks for a module with a database and an external API; .claude/skills/language-testing-patterns/SKILL.md; should"
---

I'm about to write the test suite for our Python pricing module, which talks to a database and a tax API. How should I structure the pytest fixtures and what should I mock versus run for real?
