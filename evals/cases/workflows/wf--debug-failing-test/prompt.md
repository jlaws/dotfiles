---
tags: [workflows, heavy]
max_turns: 150
timeout_seconds: 3600
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-debug python3 -m unittest fails in this repo. Find the root cause, fix it in calc/ops.py, and write ./debug-report.md with the failing test name, the exact error, the root cause, and the evidence that the fix works. When you are done, run `python3 -m unittest > ./test-output.txt 2>&1`.
