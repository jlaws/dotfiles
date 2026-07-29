# Analytics & Transformations

## dbt Model Layers

```
sources/          Raw data definitions (freshness checks mandatory)
    |
staging/          1:1 with source, light cleaning only
    |
intermediate/     Business logic, joins, aggregations (ephemeral default)
    |
marts/            Final analytics tables (dim_/fct_ prefix)
```

### Layer Rules
- **Staging**: One model per source table. Only rename, cast, lowercase. No joins. Materialized as views.
- **Intermediate**: Business logic lives here. Use `ephemeral` unless debugging.
- **Marts**: Consumer-facing. `dim_` for dimensions, `fct_` for facts. Always `table` or `incremental`.
- **Never skip layers**: Don't join sources directly in marts.

### Naming Conventions
| Layer | Prefix | Example |
|-------|--------|---------|
| Staging | `stg_<source>__<table>` | `stg_stripe__payments` |
| Intermediate | `int_<description>` | `int_payments_pivoted` |
| Marts | `dim_`/`fct_` | `dim_customers`, `fct_orders` |

## Materialization Strategy

| Materialization | When to Use |
|----------------|-------------|
| `view` | Staging models, light transforms, always-fresh data |
| `table` | Mart models <100M rows, complex transforms |
| `incremental` | Large fact tables, append-heavy event data |
| `ephemeral` | Intermediate models, CTEs that don't need their own table |

### Incremental Strategy Selection
- **`delete+insert`**: Default. Handles late-arriving data if `unique_key` set.
- **`merge`**: Rows update after initial insert. Specify `merge_update_columns`.
- **`insert_overwrite`**: Partition-based. Best for date-partitioned events on BigQuery/Spark. 3-day minimum lookback.

### Incremental Guard Rails
- Always set `unique_key` -- prevents duplicates on partial failures
- Add `on_schema_change: 'append_new_columns'` -- prevents silent column drops
- Use `{{ this }}` lookback with buffer: `where created_at > (select max(created_at) - interval '3 days' from {{ this }})`
- Run `--full-refresh` on schema changes and quarterly

## dbt Testing Strategy

### Minimum Tests Per Model
- **Staging**: `unique` + `not_null` on primary key
- **Marts**: Primary key tests + `accepted_values` on enums + `relationships` on foreign keys
- **Fact tables**: Add `dbt_utils.recency` to catch stale data

### Custom Tests Worth Writing
- Row count comparison between source and staging
- `expression_is_true` for business invariants: `total_amount >= 0`
- Freshness tests: `error_after: {count: 24, period: hour}`

### Testing Anti-Patterns
- Testing every column for `not_null` -- only test what matters
- No tests on intermediate models -- they're implementation details
- Skipping relationship tests -- broken FKs cause silent data issues

## dbt Project Organization

```
models/
  staging/
    stripe/
      _stripe__sources.yml
      _stripe__models.yml
      stg_stripe__customers.sql
  intermediate/
    finance/
      int_payments_pivoted.sql
  marts/
    core/
      _core__models.yml
      dim_customers.sql
      fct_orders.sql
```

- YAML files prefixed with `_`, named `_<source>__models.yml`
- One `sources.yml` per source system
- Group by business domain, not materialization

## dbt Macro Opinions

**Worth writing**: `cents_to_dollars(column)`, `limit_data_in_dev(column, days=3)`, `generate_schema_name` override

**Not worth writing**: Macros wrapping a single SQL function, complex Jinja harder to read than repeated SQL

## Staging Model Template

```sql
with source as (
    select * from {{ source('stripe', 'payments') }}
),

renamed as (
    select
        id as payment_id,
        lower(email) as email,
        amount / 100.0 as amount,
        created as created_at,
        _fivetran_synced as _loaded_at
    from source
)

select * from renamed
```

- Always end with `select * from <final_cte>`
- Rename to business terms in staging, not downstream
- Convert units (cents->dollars, timestamps->UTC) at staging layer

---

## SQL Query Optimization

### EXPLAIN Analysis (PostgreSQL)

```sql
EXPLAIN (ANALYZE, BUFFERS, VERBOSE) SELECT ...;
```

**Key metrics**: Seq Scan (full table, slow) | Index Scan / Index Only Scan (good/best) | Nested Loop (small joins) | Hash Join (large joins) | Merge Join (sorted data)

### Index Type Selection

| Type | Use Case |
|------|----------|
| **B-Tree** | Default: `=`, `<`, `>`, `BETWEEN`, `ORDER BY` |
| **Hash** | Equality only (`=`) |
| **GIN** | Full-text, arrays, JSONB (`@>`, `?`, `@@`) |
| **GiST** | Geometric data, ranges |
| **BRIN** | Very large naturally ordered tables (time-series) |

### Core Optimization Patterns

1. **Eliminate N+1**: Replace per-row queries with JOINs or `WHERE id IN (...)` batches
2. **Cursor pagination**: Use `WHERE (col, id) < (val, val)` instead of `OFFSET` on large tables
3. **Aggregate efficiently**: Filter before grouping, use approximate counts for UI
4. **Avoid correlated subqueries**: Replace with JOINs + aggregation or CTEs
5. **Batch writes**: Multi-row INSERT, COPY for bulk, temp table for batch UPDATE

### Materialized Views

Use for expensive aggregations queried frequently. `REFRESH CONCURRENTLY` avoids locks (requires unique index).

### Partitioning

Use `PARTITION BY RANGE` for large time-series tables. Queries auto-prune irrelevant partitions.

### Monitoring (PostgreSQL)

- **Slow queries**: `pg_stat_statements` ordered by `mean_time DESC`
- **Missing indexes**: `pg_stat_user_tables` with high `seq_scan`
- **Unused indexes**: `pg_stat_user_indexes` where `idx_scan = 0`
- **Maintenance**: `ANALYZE` for stats, `VACUUM ANALYZE` for dead tuples

> **Extended examples**: See `data/analytics-and-transformations/sql-optimization-examples.md` for full index creation patterns, batch operation templates, and monitoring queries.
