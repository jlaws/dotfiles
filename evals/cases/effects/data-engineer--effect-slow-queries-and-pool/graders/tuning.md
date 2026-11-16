---
type: llm
focus: { source: file, path: optimization.md }
---

All claims are required.
PASS if Q1 is rewritten as a half-open range on created_at (>= the day, < the next day) and its index puts status before created_at.
PASS if Q3 uses keyset pagination (a created_at/id cursor in the WHERE clause instead of OFFSET) backed by an index on user_id with created_at descending.
PASS if the pool recommendation is far below 300 (a few dozen at most, derived from the 4 cores) and the file says transaction-mode PgBouncer 1.18 does not support server-side prepared statements, so the driver must stop using them or the pool must use session mode.
FAIL if any CREATE INDEX statement lacks CONCURRENTLY, or the file tells the reader to run them inside a transaction.
