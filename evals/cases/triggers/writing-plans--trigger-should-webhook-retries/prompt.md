---
description: "a user wants a step-by-step implementation plan for a multi-component change before anyone codes; .claude/skills/writing-plans/SKILL.md; should"
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

We need webhook retries with exponential backoff in our notifier. It touches the queue worker, the config loader, and the operator docs. Write up the step-by-step implementation plan before anyone starts coding.
