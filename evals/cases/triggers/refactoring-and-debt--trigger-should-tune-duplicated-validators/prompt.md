---
description: "a user wants validation logic copy-pasted across many handlers consolidated without changing behavior; .claude/skills/refactoring-and-debt/SKILL.md; should"
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

The same email and phone validation logic is copy-pasted across nine form handlers in src/forms/, and every bug fix has to be made nine times. Clean this up without changing behavior.
