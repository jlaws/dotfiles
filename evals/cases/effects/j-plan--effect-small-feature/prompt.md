---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "write a test-first plan with exact file paths and the verification command; .claude/commands/j-plan.md; hard because a plan for a one-line fix tempts a model to skip ordering the failing test before the code change"
---

/kb:j-plan plan the fix that makes mean([]) in calc/ops.py return 0.0, as its docstring promises, written test-first. Plan mode is not available in this session, so use ./plans/plan.md as the plan file in its place; plans/ is git-ignored here, so the plan stays out of the repository. The plan must name calc/ops.py and a test file under tests/, and give python3 -m unittest as the verification command.
