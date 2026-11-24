---
type: llm
focus: { source: file, path: models/marts/fct_page_views.sql }
---

All claims are required.
PASS if fct_page_views.sql is materialized as incremental and, on incremental runs, re-reads rows from at least 3 days back (a lookback of 3 days or more, measured from the current date or from the latest event_date or event_ts already in the table).
PASS if rows re-read inside the lookback cannot be duplicated in the table: a unique_key on event_id with merge or delete+insert, or insert_overwrite of whole event_date partitions.
FAIL if the incremental filter looks back less than 3 days, for example only rows newer than the latest event_ts in the table, or a 1-day or 2-day interval.
