---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "read every doc the root instruction file names for principles or rules, never filtered by topic, and make conformance to them part of the design; .claude/commands/j-brainstorm.md; hard because the rule that decides where the setting is read sits in a principles doc whose name says nothing about display or configuration, and the request reads as asking calc to read the setting itself"
---

/kb:j-brainstorm Operators want to choose how many decimal places calc uses when it rounds a result for display, without editing code. Today nothing in calc rounds. Nothing should be implemented in this session. Work out the design and save it to ./design.md. The design must follow the project's own rules; it must compare two or three approaches and recommend one; and it must say where the operator's setting is read and how it reaches the rounding code. Do not create or edit any file other than ./design.md.
