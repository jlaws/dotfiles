---
description: "port in dependency order, leaves first and roots last, and treat dynamic dispatch as a risk that hides dependencies from a static scan; .claude/skills/code-migration/SKILL.md; hard because an order built from import statements misses the plugin that handlers.py loads by name at run time"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

We are porting the shop package in this directory from Python to TypeScript, one module at a time, with each ported module's tests passing before the next one starts. Do not write any TypeScript yet. Read the code and save ./migration-order.md with: every Python module under shop/ in the order to port them, why each one sits where it does, and the code patterns in shop/ that could make that order wrong, each cited as file:line.
