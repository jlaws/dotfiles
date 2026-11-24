---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "even with zero textual conflicts, read what main changed since the branch point and fix branch code that depends on it before force-pushing; .claude/commands/j-rebase.md; hard because main changed what pct returns without renaming it, so the rebase is clean, every test passes, and share_line prints a wrong share with no error"
---

/kb:j-rebase The pull request for feat/share (#18) is open and I'm on that branch with everything committed. gh isn't available here, so treat the open-PR check as satisfied and skip the PR text update, but do the rest. Bring the branch up to date with main as a single commit (derive the message from the branch's history) and push it to origin. What the branch adds must keep doing what it promises once it sits on main's current code. Then run `git show origin/feat/share:calc/report.py > ./pushed-report.txt` and `git rev-list --left-right --count origin/main...origin/feat/share > ./counts.txt`. Write ./report.md naming every file on the branch that depended on something main changed, whether or not you edited it.
