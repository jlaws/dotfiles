---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "branch fresh off origin/main, collapse the merged part's phases to one line citing its PR, and execute only the next part with its SHA in the ledger; .claude/commands/j-next.md; hard because a model continuing a plan tends to keep the merged phases in full and work on main"
---

/kb:j-next Continue plans/plan.md. Its first part was merged as PR #7 into origin/main (take that as given; there is no network, so do not try to reach GitHub, and stop after the local commits instead of opening a pull request). Work on a fresh branch named type/short-description, not on main, and write the branch name to ./branch.txt. Use the commit message the plan gives for the next part. In the plan, collapse the finished part's phases to one line each that cite the PR number, and tick the part you complete in the Progress checklist with its commit SHA. Finish with `git log --oneline > ./log.txt`.
