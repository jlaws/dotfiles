---
description: "a request to check a login handler for SQL injection, application code security rather than agent config; .claude/skills/config-security-audit/SKILL.md; near: security-analysis"
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Look through our Flask login handler and the user lookup queries for SQL injection and session fixation problems.
