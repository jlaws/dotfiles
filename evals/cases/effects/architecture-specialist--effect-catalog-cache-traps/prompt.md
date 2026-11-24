---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "design the cache with jittered TTLs, deploy-time warming, stampede locks or early expiry, short-TTL negative caching, version-prefixed keys, and invalidation on every source table; .claude/agents/architecture-specialist.md; hard because a page built from two tables invites invalidating on the product table alone, and negative caching and key versioning are easy to omit"
---

Use the kb:architecture-specialist agent to design a Redis cache for a product page API. Each cached page is built from the products table and the separate prices table, and the cached value format will change in future releases. Save the design to ./cache-design.md. It must handle each of these: a cold cache right after a deploy, thousands of keys written at the same moment all expiring at the same moment, a popular key expiring while 500 requests per second hit it, requests for product ids that do not exist, a price update in the prices table showing up on already-cached pages, and a release that changes the cached value format. For each one, name the specific mechanism you chose.
