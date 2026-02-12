---
name: dbt-transformation-patterns
description: dbt modeling opinions, testing strategy, and incremental patterns. Use when building data transformations, organizing dbt projects, or choosing materialization strategies.
---

# dbt Transformation Patterns

## Model Layer Opinions

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
- **Intermediate**: Business logic lives here. Use `ephemeral` unless debugging -- keeps warehouse clean.
- **Marts**: Consumer-facing. `dim_` for dimensions, `fct_` for facts. Always `table` or `incremental`.
- **Never skip layers**: Don't join sources directly in marts. The staging layer is your contract.

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
- **`delete+insert`**: Default. Simple, handles late-arriving data if unique_key set.
- **`merge`**: Use when rows update after initial insert (order status changes). Specify `merge_update_columns` to avoid full-row overwrites.
- **`insert_overwrite`**: Partition-based. Best for date-partitioned event tables on BigQuery/Spark. Always add a lookback window (3 days minimum).

### Incremental Guard Rails
- Always set `unique_key` -- without it, you get duplicates on partial failures
- Add `on_schema_change: 'append_new_columns'` -- prevents silent column drops
- Use `{{ this }}` lookback with buffer: `where created_at > (select max(created_at) - interval '3 days' from {{ this }})`
- Run `--full-refresh` on schema changes and quarterly as hygiene

## Testing Strategy

### Minimum Tests Per Model
- **Staging**: `unique` + `not_null` on primary key
- **Marts**: Primary key tests + `accepted_values` on enum columns + `relationships` on foreign keys
- **Fact tables**: Add `dbt_utils.recency` to catch stale data

### Custom Tests Worth Writing
- Row count comparison between source and staging (catch dropped data)
- `expression_is_true` for business invariants: `total_amount >= 0`
- Freshness tests on sources: `error_after: {count: 24, period: hour}`

### Testing Anti-Patterns
- Testing every column for `not_null` -- only test what matters
- No tests on intermediate models -- they're implementation details
- Skipping relationship tests -- broken foreign keys cause silent data issues

## Project Organization

```
models/
  staging/
    stripe/
      _stripe__sources.yml    # Source definitions + freshness
      _stripe__models.yml     # Model tests + docs
      stg_stripe__customers.sql
      stg_stripe__payments.sql
  intermediate/
    finance/
      int_payments_pivoted.sql
  marts/
    core/
      _core__models.yml
      dim_customers.sql
      fct_orders.sql
```

- YAML files prefixed with `_` and named `_<source>__models.yml`
- One `sources.yml` per source system
- Group by business domain, not by materialization

## Macro Opinions

### Worth Writing
- `cents_to_dollars(column)` -- used everywhere, easy to get wrong
- `limit_data_in_dev(column, days=3)` -- makes dev runs fast
- `generate_schema_name` override -- control schema naming per environment

### Not Worth Writing
- Macros that wrap a single SQL function -- just write the SQL
- Complex Jinja that's harder to read than repeated SQL

## Staging Model Template

```sql
with source as (
    select * from {{ source('stripe', 'payments') }}
),

renamed as (
    select
        id as payment_id,
        lower(email) as email,
        amount / 100.0 as amount,   -- cents to dollars in staging
        created as created_at,
        _fivetran_synced as _loaded_at
    from source
)

select * from renamed
```

- Always end with `select * from <final_cte>`
- Rename to business terms in staging, not downstream
- Convert units (cents->dollars, timestamps->UTC) at staging layer
