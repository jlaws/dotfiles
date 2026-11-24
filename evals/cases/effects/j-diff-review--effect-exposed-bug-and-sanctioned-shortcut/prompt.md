---
tags: [effects, effect, heavy]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "fix defects the diff exposes even outside it, and treat a SIMPLIFIED marker with a ceiling and upgrade path as a sanctioned decision, not a finding; .claude/commands/j-diff-review.md; hard because a reviewer flags the sort-per-call shortcut as a performance finding by habit"
---

/kb:j-diff-review feat/stats, compared against origin/main. Review the branch (it adds calc/stats.py) and fix what you find, including defects in existing code that the new code relies on; calc.ops documents that the mean of an empty list is 0.0, and variance([]) and median([]) should both return 0.0 too. Each fix is its own commit, and add tests for what you fix. gh is unavailable and there is no pull request, so do not try to open one. Write the review and your closing report to ./final-report.md: the findings with file:line and one disposition each, the verdict (PASS, CONCERNS, FAIL, or BLOCKED), the commit SHAs of your fixes, any entries you filed or left unfiled, decisions you need from me, whether another PR remains in a plan, and what happened with the PR link. Then run `python3 -m unittest > ./tests.txt 2>&1`.
