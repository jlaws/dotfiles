---
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Add structured JSON logging with a request ID to our Express middleware in src/middleware/log.js so we can trace requests across services.
