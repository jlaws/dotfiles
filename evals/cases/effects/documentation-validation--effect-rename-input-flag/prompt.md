---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Rename the --input flag of tool.py to --source, including its help text and docstring. The README usage example must run exactly as written against the new tool.py, and every mention of the old flag in the docs must be gone. Save ./change-notes.md with one line starting `Docs:` stating what you updated or N/A with a reason.
