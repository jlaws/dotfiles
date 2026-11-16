---
type: llm
focus: { source: file, path: optimization.md }
---

All claims are required.
PASS if Q1 is rewritten as a half-open range on created_at (>= the day, < the next day) and its index serves both predicates (a composite index with status leading, or a partial index on status = 'paid' over created_at).
PASS if Q3 uses keyset pagination (a created_at/id cursor in the WHERE clause instead of OFFSET) backed by an index on user_id with created_at descending.
PASS if the pool recommendation is far below 300 (a few dozen at most, derived from the 4 cores) and the file says transaction-mode PgBouncer 1.18 does not support server-side prepared statements, so the driver must stop using them, the pool must use session mode, or PgBouncer must be upgraded to a version that tracks them.
FAIL if any CREATE INDEX statement lacks CONCURRENTLY, or the file tells the reader to run them inside a transaction.
