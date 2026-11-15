---
tags: [effects, effect, heavy]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-arch design a caching layer for a read-heavy product catalog API: about 5,000 reads per second, 50 writes per second, and clients tolerate data up to 2 seconds stale. Save the design to ./design.md with three sections: cache placement, invalidation, and failure behavior.
