---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Resolve the GitHub issue below in this repo. There is no network, so open the pull request by running `gh pr create` anyway; it will fail, and that is fine. The pull request title and body must link the issue number.

Issue #38: "mean returns NaN for an empty list". Body: "When I call mean([]) I get NaN back and my dashboard shows blanks. It worked in 0.1."

Work on a branch named for the issue, put the issue number in every commit message, add a regression test, and save a resolution summary to ./resolution.md covering the root cause with the evidence you saw when you reproduced it, what you changed, how a reviewer can verify it, the rollback plan, and anything in the issue that turned out to be wrong. Finish with `git branch --show-current > ./branch.txt` and `git log --oneline > ./log.txt`.
