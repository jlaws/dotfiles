---
name: cmd-j-pr-fix
description: "Use when invoking the j-pr-fix workflow."
disable-model-invocation: true
---

# PR Fix

Invoke the `pr-comment-resolution` skill before doing anything else. Fetch all reviewer comments, categorize them, implement fixes, reply inline, verify, and push.

For complex or contested threads, you may delegate a focused re-review to the `code-reviewer` agent before replying. Verify its findings against the code.

PR: the user's provided input

If no argument provided, use `gh pr view` to identify the open PR for the current branch.
