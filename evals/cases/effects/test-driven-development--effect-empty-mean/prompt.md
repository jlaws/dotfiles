---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

mean([]) in calc/ops.py raises ZeroDivisionError, but its docstring says the mean of an empty list is 0.0. Fix it test-first: add a failing test for mean([]) in tests/test_empty.py, run it to see it fail, then fix calc/ops.py and run python3 -m unittest until everything passes.
