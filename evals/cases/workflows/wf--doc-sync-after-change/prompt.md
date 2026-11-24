---
tags: [workflows, heavy]
max_turns: 150
timeout_seconds: 3600
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "update the docs to match only the shipped flag and report each doc checked; .claude/commands/j-doc-sync.md; hard because the README names the old flag twice and a partial edit leaves one"
---

/kb:j-doc-sync this change shipped: tool.py names its output flag --output (the old --out spelling is no longer accepted). Update the documentation so README.md matches the shipped behavior and describes only the current flag, without mentioning the old spelling, and write ./doc-sync.md listing each doc you checked and what you changed.
