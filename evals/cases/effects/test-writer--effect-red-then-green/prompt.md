---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:test-writer agent to fix mean in calc/ops.py so mean([]) returns 0.0 as its docstring says, working test-first. Write the failing test in tests/test_empty.py, run `python3 -m unittest` and save its full output to ./red.txt while the test is failing, then make the minimal fix to calc/ops.py, and save the full output of the passing run to ./green.txt. Keep the existing tests in tests/test_ops.py unchanged. Report what you ran.
