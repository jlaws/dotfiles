---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-config-audit audit the agent configuration in this repo (.claude/ and .codex/). Save the report to ./config-audit.md. List each finding as `[SEVERITY] CHECK-ID: file:line - what`, grouped by severity, and end with an overall result of PASS, CONCERNS, or FAIL that reflects the most severe finding. Report a secret only where its actual value is written in a file, not where the config only references an environment variable. Never write a secret value into the report: show only where it is and what kind it is. Do not modify any config file.
