---
tags: [workflows, heavy]
max_turns: 150
timeout_seconds: 3600
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-execute-plan ./plans/plan.md. Carry the plan through to the end, including the review step that closes its unit of work. The fix lands in calc/ops.py, the new test is tests/test_empty.py, and README.md gains the sentence the plan names. When you are done, run `git log --oneline > ./log.txt`.
