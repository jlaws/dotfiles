---
tags: [workflows, heavy]
max_turns: 150
timeout_seconds: 3600
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-doc-sync this change shipped: tool.py names its output flag --output (the old --out spelling is no longer accepted). Update the documentation so README.md matches the shipped behavior, and write ./doc-sync.md listing each doc you checked and what you changed.
