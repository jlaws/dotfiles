---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Since we moved to Node 20, POST /upload returns 500 only for files over 1MB, and the log just says 'socket hang up'. I have no idea where to start looking.
