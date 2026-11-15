---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-skill-audit skills -- audit only the skills in this repo's .claude/ tree, not commands, agents, references, or config. Save the report to ./audit-report.md. Lead with the failure and warning counts from the mechanical check, list each finding with its file path and its rule id, say how to fix each, and close with the single highest-value next change. Leave out anything outside the skills scope, and do not read transcripts or memory.
