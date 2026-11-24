---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "read the root instruction file and every conventions doc it names, keep an ID-cited conventions digest, and check the design conforms to it; .claude/commands/j-brainstorm.md; hard because the conventions sit in a doc that CLAUDE.md only points to, so a model designing from the code alone misses the CalcError and no-rounding rules"
---

/kb:j-brainstorm I want calc to gain a percent_change(old, new) function. Nothing should be implemented in this session. Work out the design and save it to ./design.md. It must list each project convention that applies, by its ID, with one line on how the design conforms; it must say what percent_change does when old is 0; it must contain a numbered list of assumptions, each falsifiable; and it must compare two or three approaches and recommend one. Do not create or edit any file other than ./design.md.
