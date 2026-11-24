---
type: llm
focus: { source: file, path: models/marts/fct_page_views.sql }
---

All claims are required.
PASS if fct_page_views.sql is materialized as incremental and, on incremental runs, re-reads at least 3 days of data: at least 3 daily event_date partitions (for example the current date and the 2 dates before it), or an interval of 3 days or more measured from the current date or from the latest event_date or event_ts already in the table.
PASS if rows re-read inside the lookback cannot be duplicated in the table: a unique_key on event_id with merge or delete+insert, or insert_overwrite of whole event_date partitions.
FAIL if the incremental re-read covers fewer than 3 daily partitions, for example only rows newer than the latest event_ts in the table, or only the current date and the date before it.
