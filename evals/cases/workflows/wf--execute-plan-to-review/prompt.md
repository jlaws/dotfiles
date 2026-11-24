---
tags: [workflows, heavy]
max_turns: 150
timeout_seconds: 3600
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "execute the plan and close its unit of work with j-diff-review, whose report gives a verdict, a disposition per finding, and whether any of the plan remains; .claude/commands/j-execute-plan.md; hard because a model that finishes the tasks treats the work as done and skips the closing review"
---

/kb:j-execute-plan ./plans/plan.md. Carry the plan through to the end, including the review step that closes its unit of work. The fix lands in calc/ops.py, the new test is tests/test_empty.py, and README.md gains the sentence the plan names. Save the review step's closing report to ./review.md, with its verdict, how each finding was handled, and whether any part of the plan is left for a later pull request. When you are done, run `git log --oneline > ./log.txt`.
