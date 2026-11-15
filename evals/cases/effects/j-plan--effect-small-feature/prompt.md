---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-plan plan the fix that makes mean([]) in calc/ops.py return 0.0, as its docstring promises, written test-first. Save the plan to ./plans/plan.md. The plan must name calc/ops.py and a test file under tests/, and give python3 -m unittest as the verification command.
