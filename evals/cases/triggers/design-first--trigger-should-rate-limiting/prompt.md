---
description: "a user wants to work out how per-client rate limiting behaves before code is written; .claude/skills/design-first/SKILL.md; should"
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Before anyone writes code, I want to think through how per-client rate limiting should behave on our public REST API: what the limits key on, what happens at the boundary, and how clients learn they were throttled. Help me work out the behavior first.
