set -euo pipefail
mkdir -p docs models/staging models/marts
cat > dbt_project.yml <<'YML'
name: analytics
version: "1.0.0"
profile: analytics
model-paths: ["models"]
YML
cat > models/staging/sources.yml <<'YML'
version: 2
sources:
  - name: raw
    tables:
      - name: events
YML
cat > models/staging/stg_events.sql <<'SQL'
select
    event_id,
    user_id,
    page_path,
    cast(event_ts as timestamp) as event_ts,
    date(event_ts) as event_date
from {{ source('raw', 'events') }}
SQL
cat > models/marts/fct_page_views.sql <<'SQL'
{{ config(
    materialized='table',
    partition_by={'field': 'event_date', 'data_type': 'date'}
) }}

select
    event_id,
    user_id,
    page_path,
    event_ts,
    event_date
from {{ ref('stg_events') }}
SQL
cat > docs/events.md <<'MD'
# Page view events

- Warehouse: BigQuery. `raw.events` is partitioned by day on `event_ts` and grows by about
  400 million rows a day.
- The mobile apps queue events while offline, so an event can reach `raw.events` up to 24 hours
  after its `event_ts`.
- `event_id` is unique per event.
- `fct_page_views` is rebuilt in full every hour today. That scan is the largest line on the
  BigQuery bill.
MD
