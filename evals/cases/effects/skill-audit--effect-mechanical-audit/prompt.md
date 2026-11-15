---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Audit the .claude tree in this repository with the integrity check the knowledge base provides. Save the results to ./audit.md with: the exit status the check returned, each failing asset with its file path and the reason it fails, and the warnings grouped separately from the failures. Report what the check printed; do not re-derive findings by hand.
