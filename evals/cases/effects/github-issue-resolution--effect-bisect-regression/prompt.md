---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "in root cause analysis, find the commit that introduced the regression and back it with evidence such as bisect output or a test run at that commit and its parent; .claude/skills/github-issue-resolution/SKILL.md; hard because git blame on the failing line points at a later formatting commit, so a model that trusts the blame without testing the history names the wrong commit"
---

Resolve the GitHub issue below in this repo. There is no network, so open the pull request by running `gh pr create` anyway; it will fail, and that is fine. The pull request title and body must link the issue number.

Issue #61: "mean([]) crashes again". Body: "On main, mean([]) raises ZeroDivisionError. It returned 0.0 in v0.2."

Work on a branch named for the issue, put the issue number in every commit message, and add a regression test. Save a resolution summary to ./resolution.md that names the commit that introduced the regression, with the evidence that pins it to that commit, and covers the root cause, what you changed, how a reviewer can verify it, and the rollback plan.
