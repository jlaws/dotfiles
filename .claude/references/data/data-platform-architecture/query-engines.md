# Query Engine Integration

Connecting analytic query engines to lakehouse table formats. Each pairing has distinct performance characteristics and use cases.

## Query Engine Selection

| Engine | Best For | Latency | Ecosystem | Catalog Support |
|--------|----------|---------|-----------|-----------------|
| **DuckDB** | Single-machine OLAP, dev/test, data pipelines | <1s (in-process) | Python, SQL, Arrow | Iceberg, Hudi, Delta (limited) |
| **Trino** | Distributed SQL, federated queries, analytics | 1-10s | Multi-engine, BI tools | Iceberg, Delta, Hudi, S3, Postgres |
| **Presto** | Large-scale queries, interactive dashboards | 1-10s | Wide integration | All table formats via connectors |
| **Spark SQL** | Unified batch+interactive, ML pipelines | 5-30s | Hadoop ecosystem, MLlib | Delta (native), Iceberg, Hudi |

---

## DuckDB with Iceberg

Single-machine, in-process OLAP engine. Excellent for local development and small-to-medium analytics.

```python
import duckdb

con = duckdb.connect()
con.install_extension("iceberg")
con.load_extension("iceberg")

# Direct Iceberg scan
df = con.sql("""
    SELECT user_id, count(*) as events, sum(amount) as total
    FROM iceberg_scan('s3://lakehouse/analytics/events')
    WHERE timestamp >= '2025-01-01'
    GROUP BY user_id ORDER BY total DESC LIMIT 100
""").fetchdf()

# Time travel via snapshot
historical = con.sql("""
    SELECT * FROM iceberg_scan('s3://lakehouse/events', snapshot_id=123456789)
    WHERE event_date = '2024-12-01'
""").fetchdf()
```

**Strengths**: No server overhead, native Arrow support, immediate results. **Limits**: Single-machine memory; not suitable for >100GB data without streaming.

---

## DuckDB with Hudi

DuckDB's Hudi integration focuses on incremental reads and CDC (Change Data Capture).

```python
import duckdb

con = duckdb.connect()
con.install_extension("hudi")
con.load_extension("hudi")

# Read Hudi table with CDC
df = con.sql("""
    SELECT _hoodie_record_key, _hoodie_commit_time, col1, col2
    FROM 's3://lakehouse/hudi_table'
    WHERE _hoodie_commit_time >= '20250101000000'
""").fetchdf()
```

---

## Trino with Iceberg

Distributed SQL query engine. Best for federated analytics across multiple data sources and concurrent users.

```sql
-- Standard query against Iceberg catalog
SELECT date_trunc('hour', event_time) AS hour,
       count(*) AS events,
       approx_percentile(latency_ms, 0.99) AS p99
FROM iceberg.analytics.api_events
WHERE event_time >= current_date - INTERVAL '7' DAY
GROUP BY 1 ORDER BY 1;

-- Time travel via snapshot
SELECT * FROM iceberg.analytics.events FOR VERSION AS OF 123456789
WHERE date = '2024-12-01';

-- Federated query: Iceberg + PostgreSQL
SELECT t.user_id, t.order_total, u.email
FROM iceberg.analytics.orders t
JOIN postgres.public.users u ON t.user_id = u.id
WHERE t.order_date >= DATE '2025-01-01';
```

**Strengths**: Distributed query, multi-catalog, SQL standard compliance. **Overhead**: Server management, network latency.

---

## Trino with Delta Lake

Delta Lake support via Trino connector. Less native than Iceberg but improving.

```sql
-- Query Delta table via Trino
SELECT date, region, sum(amount) as total
FROM delta.default.sales_data
WHERE date >= DATE '2025-01-01'
GROUP BY 1, 2;

-- Note: Time travel (AS OF) has limited support in Delta via Trino
-- Use Delta-native tools (Spark) for point-in-time queries
```

---

## Performance Comparison

| Operation | DuckDB | Trino | Spark SQL |
|-----------|--------|-------|-----------|
| Scan 10GB Parquet (local disk) | ~2s | N/A | ~5s |
| Scan 10GB Iceberg (S3) | ~10s | ~30s | ~20s |
| Join 5GB + 500MB | ~3s | ~8s | ~10s |
| Aggregation (GROUP BY) | ~1s (memory) | ~5s | ~8s |
| Concurrent queries (10 users) | No | Yes (via queue) | Limited (cluster-shared) |

---

## Gotchas and Gotcha Mitigation

- **Catalog lock contention**: Trino/Presto may block on Iceberg metadata ops under high write concurrency; use per-partition staging tables
- **Network latency S3**: DuckDB/Trino both hit S3 latency (~100ms per request); local caching via S3 Select helps for repeated queries
- **Schema evolution mismatch**: If table schema changes between DuckDB and Trino queries, results may differ; always validate schema versions
- **Memory explosion in joins**: Broadcast joins in Trino require all lookup data in memory; watch executor memory for large dimensions
- **Snapshot ID portability**: Snapshot IDs are catalog-specific; time travel via snapshot ID works only in same catalog instance

---

## Integration Patterns

### Interactive Analysis → Batch Production
```
DuckDB (local notebook)  →  Trino (shared warehouse)  →  Spark (scheduled pipeline)
```
Develop in DuckDB for instant feedback, scale to Trino for team queries, automate with Spark pipelines.

### Multi-Catalog Federated Analytics
```sql
-- Trino: join Iceberg + Delta tables from different catalogs
SELECT i.user_id, d.order_id
FROM iceberg_catalog.analytics.users i
JOIN delta_catalog.warehouse.orders d ON i.user_id = d.user_id;
```
