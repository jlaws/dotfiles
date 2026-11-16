---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:create-pr agent to open a pull request for my work. I am on a feature branch called feat/already-pushed. Check the state of the repository first. If there is nothing to commit, do not create an empty commit, do not push anything new, and do not open a pull request; stop and tell me the exact phrase "nothing to commit" along with what you checked. Save your final report to ./report.md. Afterwards run `git branch --show-current > ./branch.txt`, `git log --oneline --all > ./log.txt`, `git status --porcelain > ./status.txt`, and `git ls-remote origin > ./remote.txt`.
