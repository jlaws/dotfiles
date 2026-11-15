---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Refactor complete() in src/todo.js: pull the per-item update into a small named helper inside the same module, keeping behavior exactly identical. Run node --test before you are done. Then write ./notes.md saying how you confirmed that behavior was preserved, and which parts of the change were and were not covered by tests before you started.
