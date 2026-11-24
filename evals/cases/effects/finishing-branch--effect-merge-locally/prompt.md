---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "when the user names merge locally, update the base from origin before merging, verify tests on the merged result, then delete the branch; .claude/skills/finishing-branch/SKILL.md; hard because local main is one commit behind origin, and a merge into the stale local main looks complete while dropping the upstream commit"
---

I'm on feat/readme-rst and the work is finished. This one doesn't need review: merge it into main here, with no pull request; main gets pushed to origin later, and that push has to be a plain fast-forward. Then run `python3 -m unittest > ./merged-tests.txt 2>&1` on main, delete the finished branch, and save `git branch --list > ./branches.txt` and `git log --oneline --graph > ./log.txt` from main afterwards.
