---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-config-audit check only the permissions in this repo's agent config (.claude/ and .codex/): over-broad allow-lists, agent tool grants, and auto-approved destructive or remote-execution commands. Save the report to ./permissions-audit.md. Do not report on secrets or prompt injection at all, and do not repeat any credential in the file. List each finding as `[SEVERITY] CHECK-ID: file:line - what`. Do not modify any config file.
