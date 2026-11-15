---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-data plan a zero-downtime change on PostgreSQL 15. The table events has 300 million rows and takes writes all day. Add a NOT NULL column public_id uuid that defaults to gen_random_uuid() for new rows, and add a unique index on it. Save the migration plan to ./migration.md as numbered steps with the exact SQL for each step, and state which steps must run outside a transaction and why.
