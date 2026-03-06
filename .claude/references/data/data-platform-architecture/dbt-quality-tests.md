# dbt Data Quality Tests

## Built-in + dbt-utils Tests

```yaml
# models/marts/core/_core__models.yml
version: 2
models:
  - name: fct_orders
    tests:
      - dbt_utils.recency:
          datepart: day
          field: created_at
          interval: 1
    columns:
      - name: order_id
        tests: [unique, not_null]
      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('dim_customers')
              field: customer_id
      - name: order_status
        tests:
          - accepted_values:
              values: ['pending', 'processing', 'shipped', 'delivered', 'cancelled']
      - name: total_amount
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0"
```

## Custom Generic Test

```sql
-- tests/generic/test_row_count_in_range.sql
{% test row_count_in_range(model, min_count, max_count) %}
with row_count as (select count(*) as cnt from {{ model }})
select cnt from row_count
where cnt < {{ min_count }} or cnt > {{ max_count }}
{% endtest %}
```

## Custom Singular Test

```sql
-- tests/singular/assert_orders_customers_match.sql
with orphaned_orders as (
    select o.customer_id
    from (select distinct customer_id from {{ ref('fct_orders') }}) o
    left join {{ ref('dim_customers') }} c using (customer_id)
    where c.customer_id is null
)
select * from orphaned_orders
```
