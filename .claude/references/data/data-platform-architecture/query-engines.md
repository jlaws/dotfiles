# Query Engine Integration

## DuckDB with Iceberg

```python
import duckdb

con = duckdb.connect()
con.install_extension("iceberg")
con.load_extension("iceberg")

df = con.sql("""
    SELECT user_id, count(*) as events, sum(amount) as total
    FROM iceberg_scan('s3://lakehouse/analytics/events')
    WHERE timestamp >= '2025-01-01'
    GROUP BY user_id ORDER BY total DESC LIMIT 100
""").fetchdf()
```

## Trino with Iceberg

```sql
-- Query against Iceberg catalog
SELECT date_trunc('hour', event_time) AS hour,
       count(*) AS events,
       approx_percentile(latency_ms, 0.99) AS p99
FROM iceberg.analytics.api_events
WHERE event_time >= current_date - INTERVAL '7' DAY
GROUP BY 1 ORDER BY 1;

-- Time travel
SELECT * FROM iceberg.analytics.events FOR VERSION AS OF 123456789;
```
