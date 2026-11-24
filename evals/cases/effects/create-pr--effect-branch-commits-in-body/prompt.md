---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "on a feature branch, commit the pending work on that branch and derive the PR body bullets from git log main..branch, so the body covers every commit on the branch; .claude/agents/create-pr.md; hard because a model summarizes the diff it just staged and leaves out the commit already on the branch"
---

Use the kb:create-pr agent to open a pull request for my work on this branch. I did not choose a commit message, so derive one from the changes. Push to origin and attempt the pull request with gh (it is not signed in here, so it may fail; still attempt it). Save the PR title and body to ./pr-body.md, with bullet points of what changed in this pull request, a Test Plan naming the verification you ran, and a Merge Danger section (one-way or two-way door, blast radius in one word, how to revert). Afterwards run `git log --oneline --all > ./log.txt` and `git --git-dir=.remote.git for-each-ref --format='%(objectname)%09%(refname)' refs/heads > ./remote.txt`.
