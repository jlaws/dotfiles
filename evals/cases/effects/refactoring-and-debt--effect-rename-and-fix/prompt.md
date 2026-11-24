---
description: "keep a refactoring and a behavior change in separate commits; .claude/skills/refactoring-and-debt/SKILL.md; hard because a task that asks for both at once invites one commit that renames and fixes together"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

In this repo, rename mean to average in calc/ops.py and update every caller and test. Separately, average([]) must return 0.0 as the docstring promises, because today it raises ZeroDivisionError. Commit your work in git as you go, and save ./log.md with one line per commit: its message and `behavior-change: yes` or `behavior-change: no`. Run python3 -m unittest and make sure it passes at the end.
