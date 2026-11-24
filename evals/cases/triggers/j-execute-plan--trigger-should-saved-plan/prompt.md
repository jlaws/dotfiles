---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "a user with a saved phased plan asks to carry it out task by task with verification; .claude/commands/j-execute-plan.md; should"
---

The plan for moving our logging to structured JSON is saved at ./plans/structured-logging.md (three phases, each with a verification command). Carry it out task by task and run each verification before you move on.
