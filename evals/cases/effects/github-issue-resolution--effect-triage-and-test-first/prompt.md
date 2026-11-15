---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Resolve the GitHub issue below in this repo. There is no network, so open the pull request by running `gh pr create` anyway; it will fail, and that is fine.

Issue #57: "complete() silently ignores unknown ids". Body: "Our UI calls complete(list, id) after a stale click and shows a success toast, but nothing changed because the id no longer exists. It should throw an Error whose message says the todo was not found."

Triage first: classify the issue P0, P1, P2 or P3 with one sentence of reasoning, at the top of ./resolution.md. Write the failing test before changing src/todo.js and show it failing. Put the test and the fix in separate commits that each reference the issue number, on a branch named for the issue. The pull request title must start with "Fix #57" and its body must say it fixes #57. Finish the summary in ./resolution.md with the verification steps for a reviewer, and run `git log --oneline > ./log.txt`.
