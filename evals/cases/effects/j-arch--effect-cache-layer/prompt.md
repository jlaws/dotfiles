---
tags: [effects, effect, heavy]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "design a cache layer from the architecture references with invalidation that keeps staleness inside the stated bound and a defined behavior when the cache is down; .claude/commands/j-arch.md; hard because a cache-aside design with a default TTL looks complete while breaking the 2-second staleness bound or leaving cache failure unhandled"
---

/kb:j-arch design a caching layer for a read-heavy product catalog API: about 5,000 reads per second, 50 writes per second, and clients tolerate data up to 2 seconds stale. Save the design to ./design.md with three sections: cache placement, invalidation, and failure behavior, and show how the design handles the stated read and write rates.
