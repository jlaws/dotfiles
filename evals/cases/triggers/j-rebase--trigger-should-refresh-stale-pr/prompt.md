---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "a user with a messy open PR behind main asks to squash, rebase, verify, and force-push; .claude/commands/j-rebase.md; should"
---

My open PR on the feat/export branch has seven messy commits and main has moved a lot since I branched. Squash it down to one commit, rebase it on the latest main, check the conflict resolutions still work, and force-push so the PR is current.
