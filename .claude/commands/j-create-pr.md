---
name: j-create-pr
description: "Automate PR workflow — stage, commit, branch, push, and open a GitHub PR. Use when ready to submit changes for review. Do NOT use if changes are incomplete (finish implementation first)."
argument-hint: "<description of changes>"
model: sonnet
effort: low
---

Invoke the create-pr agent to handle the full PR workflow:

Stage, commit, branch, push, and open a GitHub PR for the current changes. Description of changes: $ARGUMENTS
