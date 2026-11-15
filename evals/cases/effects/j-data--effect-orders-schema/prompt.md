---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-data write a PostgreSQL 16 schema to ./schema.sql for an orders system: customers (unique email, optional phone), orders (belongs to a customer, status, placed-at time, total), and order_items (belongs to an order, product sku, quantity, unit price). Requirements: auto-generated integer keys use identity columns rather than serial; money uses an exact numeric type; every point-in-time column keeps the time zone; each foreign key column has its own index; text columns carry no length limits.
