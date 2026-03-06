---
name: create-pr
description: "Automate PR workflow — stage, commit, branch, push, and open a GitHub PR. Use when ready to submit changes for review."
argument-hint: "<description of changes>"
---

Use the Agent tool with subagent_type "create-pr" to handle the full PR workflow. Pass the following as the agent prompt:

Stage, commit, branch, push, and open a GitHub PR for the current changes. Description of changes: $ARGUMENTS
