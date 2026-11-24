---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "splitting a service out of a monolith and choosing sync or async calls and API versioning; .claude/commands/j-arch.md; should"
---

We are carving the order service out of our Django monolith. Should the payment calls between them be synchronous REST or async events, and how should we version the public API so partners are not broken? I want a recommendation with the trade-offs.
