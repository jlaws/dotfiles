---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "on main with no message given, derive a `type/short-description` branch and an imperative commit, stage only the real change, warn on credentials, push, and write a PR body with bullets, a Test Plan, and a Merge Danger section; .claude/agents/create-pr.md; hard because staging everything sweeps in credentials and scratch files, and the Merge Danger section is a house format"
---

Use the kb:create-pr agent to open a pull request for my uncommitted work on main. I did not choose a branch name or a commit message, so derive both from the changes: the commit message in imperative mood, under 72 characters, no trailing period; the branch name a short kebab-case description of the change (any prefix is fine, but not main). Stage only the real change (the README and calc/ops.py edits) and leave my scratch notes and credentials.json out of the commit, but tell me you saw credentials.json. Push to origin and attempt the pull request with gh (it is not signed in here, so it may fail; still attempt it). Save the PR title and body to ./pr-body.md, with bullet points of what changed, a Test Plan naming the verification you ran, and a Merge Danger section (one-way or two-way door, blast radius in one word, how to revert). Afterwards run `git branch --show-current > ./branch.txt`, `git log --oneline --all > ./log.txt`, `git diff --name-only origin/main HEAD > ./changed.txt`, `git status --porcelain > ./status.txt`, and `git --git-dir=.remote.git for-each-ref --format='%(objectname)%09%(refname)' refs/heads > ./remote.txt`.
