---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "tune PostgreSQL queries with half-open date ranges, an expression index on lower(email), keyset pagination, and concurrent index builds, size the pool from cores, and stop server-side prepared statements through transaction-mode PgBouncer; .claude/agents/data-engineer.md; hard because a pool of hundreds looks safe and transaction pooling silently breaks prepared statements"
---

Use the kb:data-engineer agent to tune the three slow queries in ./slow.sql against the schema in ./schema.sql. Both tables are in production and take writes all day, so every index must be buildable without blocking writes. Save ./optimization.md with, for each query, the rewritten SQL and the exact CREATE INDEX statement it needs. Q3 must stay fast for deep pages of one user's history. Also answer in the same file: the application keeps a pool of 300 connections to this 4-core, SSD-backed database through PgBouncer 1.18 in transaction mode, and the application driver uses server-side prepared statements. Recommend a pool size with your reasoning, and say what must change for the driver and PgBouncer to work together.
