---
description: "a user hands over a three-task plan and wants each task done by a separate agent and checked by another; .claude/skills/subagent-driven-development/SKILL.md; should"
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Here's the plan: Task 1, add a rate_limit setting to config.yaml and load it in settings.py. Task 2, enforce it in the API middleware. Task 3, document it in docs/limits.md. Hand each task to a separate agent, and have a different agent check each one before moving on.
