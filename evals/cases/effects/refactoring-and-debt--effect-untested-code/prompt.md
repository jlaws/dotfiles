---
description: "before a structural change to code with no test coverage, write tests that pin its current behavior; .claude/skills/refactoring-and-debt/SKILL.md; hard because the module already has a test file, so a model can assume complete() is covered and refactor first"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Refactor complete() in src/todo.js: pull the per-item update into a small named helper inside the same module, keeping behavior exactly identical. Show the behavior was preserved with tests that exercise complete(), in test/todo.test.js, in place before the structural change. Run node --test before you are done. Then write ./notes.md saying how you confirmed that behavior was preserved, and which parts of the change were and were not covered by tests before you started.
