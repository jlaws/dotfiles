---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

I'm on feat/readme-rst and the work is finished. This one doesn't need review: merge it into main here, with no pull request. Then run `python3 -m unittest > ./merged-tests.txt 2>&1` on main, delete the finished branch, and save `git branch --list > ./branches.txt` and `git log --oneline --graph > ./log.txt` from main afterwards.
