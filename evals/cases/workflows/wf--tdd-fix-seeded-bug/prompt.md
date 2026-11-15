---
tags: [workflows, heavy]
max_turns: 150
timeout_seconds: 3600
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-tdd make mean([]) in calc/ops.py return 0.0, as its docstring says. Write the failing test first in tests/test_empty.py, watch it fail, then fix calc/ops.py, run python3 -m unittest until it passes, and commit. When you are done, run `python3 -m unittest > ./test-output.txt 2>&1`.
