---
tags: [workflows, heavy]
max_turns: 150
timeout_seconds: 3600
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "write the failing test first and watch it fail, then fix and commit; .claude/commands/j-tdd.md; hard because the fix is obvious, so a model edits calc/ops.py before writing the test"
---

/kb:j-tdd make mean([]) in calc/ops.py return 0.0, as its docstring says. Write the failing test first in tests/test_empty.py, watch it fail, then fix calc/ops.py, run python3 -m unittest until it passes, and commit. When you are done, run `python3 -m unittest > ./test-output.txt 2>&1`.
