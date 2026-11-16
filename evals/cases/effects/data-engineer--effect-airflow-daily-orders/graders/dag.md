---
type: llm
focus: { source: file, path: dags/daily_orders_fixed.py }
---

All claims are required.
PASS if the load is idempotent for a day: it deletes that day's rows before inserting in one transaction, or uses an upsert or MERGE keyed on the order, so a re-run does not duplicate rows.
PASS if the loaded date comes from the run (a Jinja {{ ds }} template, the logical date, or data_interval_start) and no date literal remains in the SQL.
PASS if no psycopg2.connect call or SQL execution happens at module level; the connection is opened inside the task callable.
FAIL if depends_on_past is still True, or the sensor or load task has no timeout (execution_timeout, or timeout on the sensor).
