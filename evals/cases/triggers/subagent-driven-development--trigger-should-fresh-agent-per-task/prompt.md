---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Run through the saved plan at plans/billing-migration.md, but give each task to its own fresh agent and have every task reviewed by a separate agent before the next one starts.
