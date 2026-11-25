---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "inside a plan that owns the commit, apply the doc edits and commit them instead of stopping for approval; .claude/skills/post-ship-doc-sync/SKILL.md; hard because a model told never to auto-commit doc changes stops at a preview even though the plan's PR is the review point"
---

You are executing phase 2 of an approved plan. Phase 1 shipped release v0.2 of this tool, which renamed the --out flag to --output (the code already has the change). The plan's PR is reviewed before merge, so this phase runs without pausing for approval.

PLAN
Phase 2: Docs
- Task 1: Sync README.md to the shipped code, changing only the lines the rename requires.
- Task 2: Commit the edit as one commit with the message "Sync README to the --output rename".

Finish with `git log --oneline > ./log.txt` and `git status --porcelain > ./status.txt`.
