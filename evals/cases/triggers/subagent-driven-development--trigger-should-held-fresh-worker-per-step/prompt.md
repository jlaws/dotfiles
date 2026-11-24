---
description: "a user wants a fresh worker per plan step with each step reviewed before the next begins; .claude/skills/subagent-driven-development/SKILL.md; should"
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Work through plans/auth-cleanup.md with a fresh worker for each step, and review each step's output before the next one begins.
