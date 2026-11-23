---
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Convert this one query from MySQL to Postgres syntax: SELECT IFNULL(total, 0) FROM orders LIMIT 5, 10;
