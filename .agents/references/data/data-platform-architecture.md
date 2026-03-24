# Data Platform Architecture

## Platform Component Selection

| Need | Component | Key Decision |
|------|-----------|-------------|
| Storage + table format | Lakehouse | Iceberg vs Delta vs Hudi (see below) |
| Data validation | Quality framework | Great Expectations vs dbt tests vs SodaCL |
| ML feature serving | Feature store | Feast vs Tecton vs custom (Redis + warehouse) |
| Data contracts | Schema governance | datacontract.com spec + SodaCL checks |
| File maintenance | Compaction / vacuum | Schedule based on write frequency |

## Lakehouse Storage

### Table Format Decision

| Criteria | Apache Iceberg | Delta Lake | Apache Hudi |
|----------|---------------|------------|-------------|
| Multi-engine support | Best (Spark, Trino, Flink, DuckDB) | Good (Spark-native, growing) | Good (Spark, Flink) |
| Schema evolution | Full (add, drop, rename, reorder) | Add/rename columns | Add columns |
| Hidden partitioning | Yes (no partition columns in queries) | No (explicit partition cols) | No |
| Partition evolution | Yes (change without rewrite) | No (requires rewrite) | No |
| Time travel | Snapshot-based, configurable | Version-based, 30-day default | Timeline-based |
| CDC / upserts | Merge-on-read or copy-on-write | MERGE INTO | Built-in (MoR native) |
| Best for | Multi-engine lakehouse | Databricks-centric orgs | CDC-heavy workloads |

### Iceberg Core Operations

Schema evolution, partition evolution, time travel -- see `references/iceberg-patterns.py`.

Key points:
- **Hidden partitioning**: users write `timestamp`, Iceberg partitions by day automatically
- **Partition evolution**: change strategy without rewriting existing data
- **Schema evolution**: add, drop, rename, reorder columns -- no data rewrite needed

### Delta Lake Core Operations

MERGE INTO (upsert), OPTIMIZE, VACUUM, Change Data Feed -- see `references/delta-patterns.py`.

Key points:
- **MERGE INTO**: full CDC with match/not-matched/not-matched-by-source
- **Z-Order**: multi-dimensional clustering for 2-4 columns (diminishing returns beyond)
- **Change Data Feed**: track insert/update/delete for downstream consumers

### Query Engine Integration

DuckDB + Iceberg and Trino + Iceberg patterns -- see `references/query-engines.md`.

### Compaction and Maintenance

| Task | Iceberg | Delta Lake |
|------|---------|------------|
| Compact small files | `rewrite_data_files` | `OPTIMIZE` / `executeCompaction` |
| Clustering | Sort via rewrite | Z-Order |
| Remove old versions | `expire_snapshots` | `VACUUM` (min 7 days retention) |
| Orphan cleanup | `remove_orphan_files` | Automatic with VACUUM |

### Lakehouse Gotchas

- **Small file problem** -- frequent writes create thousands of tiny files; schedule compaction
- **VACUUM too aggressive** -- retention below 7 days (Delta) risks breaking concurrent readers
- **Partition cardinality** -- >10K partitions degrades metadata; use bucket transforms
- **Schema evolution with Parquet** -- column renames work in Iceberg (name-based) but break raw Parquet (position-based)
- **Catalog lock contention** -- concurrent writers need atomic commits; use catalog-level locking
- **Cost of time travel** -- every snapshot retains file references; unbounded snapshots bloat metadata
- **Delta outside Databricks** -- UniForm improves compatibility but some features are Databricks-only
- **Partition evolution pitfall** -- old data keeps old layout; spanning queries may scan more files

## Data Quality

### Quality Dimensions

| Dimension | Example Check | Framework |
|-----------|---------------|-----------|
| Completeness | `not_be_null` | GE / dbt `not_null` |
| Uniqueness | `to_be_unique` | GE / dbt `unique` |
| Validity | `to_be_in_set` | GE / dbt `accepted_values` |
| Accuracy | Cross-reference validation | Custom / singular tests |
| Consistency | Column-pair comparisons | GE / dbt `expression_is_true` |
| Timeliness | Freshness check | dbt `recency` / SodaCL `freshness` |

### Framework Selection

| Criteria | Great Expectations | dbt Tests | SodaCL |
|----------|-------------------|-----------|--------|
| Integration | Standalone / Airflow | dbt-native | Standalone / Airflow |
| Test authoring | Python API | YAML + SQL | YAML (SodaCL DSL) |
| Data docs | Built-in HTML reports | None (use Elementary) | Soda Cloud |
| Best for | Complex validation logic | Transform-layer testing | Contract enforcement |

### Great Expectations Suite

Programmatic suite building and checkpoint configuration -- see `references/great-expectations.py`.

Key pattern: build expectation suites covering schema, primary keys, categorical values, numeric ranges, and row counts.

### dbt Data Tests

Built-in + custom generic and singular tests -- see `references/dbt-quality-tests.md`.

Key pattern: combine `unique`, `not_null`, `relationships`, `accepted_values` with custom `expression_is_true` and `recency` tests.

### Data Contracts

```yaml
apiVersion: datacontract.com/v1.0.0
kind: DataContract
metadata:
  name: orders
  version: 1.0.0
  owner: data-platform-team
schema:
  type: object
  properties:
    order_id: { type: string, format: uuid, required: true, unique: true }
    customer_id: { type: string, format: uuid, required: true, pii: true }
    total_amount: { type: number, minimum: 0, maximum: 100000 }
    status: { type: string, enum: [pending, processing, shipped, delivered, cancelled] }
quality:
  type: SodaCL
  specification:
    checks for orders:
      - row_count > 0
      - missing_count(order_id) = 0
      - duplicate_count(order_id) = 0
      - freshness(created_at) < 24h
sla:
  availability: 99.9%
  freshness: 1 hour
  latency: 5 minutes
```

### Automated Quality Pipeline

Orchestrate validation across tables with result aggregation -- see `references/great-expectations.py` (QualityPipeline class).

## Feature Store

### Architecture Decision

| Criteria | Feast (OSS) | Tecton | Custom (Redis + Warehouse) |
|----------|------------|--------|---------------------------|
| Setup cost | Low | High (SaaS) | Medium-High |
| Online serving latency | <10ms (Redis) | <5ms | Depends on impl |
| Streaming features | Limited (push-based) | Native Spark/Flink | Build your own |
| Point-in-time joins | Built-in | Built-in | Must implement |
| Team size sweet spot | 2-15 | 15-100+ | 5-20 (eng-heavy) |

**Recommendation**: Feast for most teams -- covers 80% of use cases with minimal operational burden.

### Feast Patterns

Feature definitions, point-in-time joins, online serving, streaming push -- see `references/feast-patterns.py`.

Key concepts:
- **Entities**: natural grain keys (`user_id`, `product_id`)
- **BatchFeatureView**: offline-computed features materialized to online store
- **Point-in-time join**: prevents future data leaking into training examples
- **Materialization**: `feast materialize-incremental` syncs offline -> online

### Feature Store Gotchas

| Problem | Symptom | Fix |
|---------|---------|-----|
| Training-serving skew | Model degrades in prod | Define transforms once; compare online vs offline values |
| Time-travel bugs | Backfilled data pollutes history | Use business timestamp, not ingestion timestamp |
| Feature freshness | Stale predictions | Monitor feature age; alert when exceeding TTL |
| Entity key bloat | Slow lookups, high memory | One entity per natural grain; aggressive TTL on session-level |

### Entity Key Design

| Entity Pattern | Online Store Size | Lookup Speed | Use Case |
|---------------|------------------|-------------|----------|
| `user_id` | ~N users | Fast | User-level aggregates |
| `product_id` | ~N products | Fast | Product metadata/stats |
| `(user_id, product_id)` | ~N*M | Slow if M large | Interaction features |
| `session_id` | Unbounded | Degrades | Avoid; use TTL aggressively |
