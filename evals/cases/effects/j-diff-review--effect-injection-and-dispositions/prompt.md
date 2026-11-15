---
tags: [effects, effect, heavy]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-diff-review feat/user-lookup, compared against origin/main. Review the branch (it adds app/users.py) and fix what you find, with each fix as its own commit. gh is unavailable and there is no pull request yet, so do not try to open one; say so at the end of the report. Save the full review to ./review.md with severity sections (Critical, High, Medium) plus Test Gaps, Doc Gaps, Observability Gaps and What Looks Good, every finding citing file:line and carrying exactly one disposition (fixed, filed, or asked), and a closing verdict of PASS, CONCERNS, FAIL, or BLOCKED that reflects what you found. Afterwards run `git log --oneline origin/main..HEAD > ./log.txt`.
