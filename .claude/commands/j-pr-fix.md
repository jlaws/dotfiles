---
name: j-pr-fix
description: "Resolve all PR reviewer comments — categorize, fix, reply inline, verify, and push. Use when you have open PR review comments to address. Do NOT use if there are no comments yet (wait for review)."
argument-hint: "<pr-number-or-url>"
---

Invoke the `pr-comment-resolution` skill via the Skill tool before doing anything else. Fetch all reviewer comments, categorize them, implement fixes, reply inline, verify, and push.

PR: $ARGUMENTS

If no argument provided, use `gh pr view` to identify the open PR for the current branch.
