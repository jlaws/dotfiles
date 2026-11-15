---
type: llm
focus: { source: file, path: migration.md }
---

All claims are required.
PASS if migration.md does not add the column as NOT NULL with the gen_random_uuid() default in a single ALTER TABLE on the populated table.
PASS if the column is added nullable first, existing rows are backfilled in batches, and NOT NULL is enforced afterward (for example via a validated CHECK constraint, then SET NOT NULL).
PASS if the plan says CREATE INDEX CONCURRENTLY cannot run inside a transaction block.
FAIL if the backfill is a single UPDATE over all 300 million rows.
