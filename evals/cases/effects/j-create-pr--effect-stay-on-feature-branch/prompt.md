---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "stay on the current non-main branch and push it with its earlier unpushed commit rather than branching again; .claude/commands/j-create-pr.md; hard because a PR workflow habitually cuts a fresh branch, which would strand the unpushed commit"
---

/kb:j-create-pr Add subtract tests. I'm on feat/subtract, which already has one local commit ("Add subtract") that was never pushed, and tests/test_subtract.py is new and uncommitted. Stay on this branch (do not create another branch), commit the test file using the sentence above as the commit message, push the branch to origin, and open a pull request; gh is not signed in here, so still attempt the pull request with gh even if it fails. The pull request must carry both commits and local main must not move. Afterwards run `git branch -a > ./branches.txt`, `git log --format=%s origin/main..origin/feat/subtract > ./commits.txt`, and `git diff --name-only origin/main origin/feat/subtract > ./changed.txt`.
