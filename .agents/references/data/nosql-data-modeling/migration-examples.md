# Relational to NoSQL Migration Examples

Side-by-side SQL and NoSQL query comparisons showing how relational patterns map to document/key-value models.

## Relational to MongoDB

```sql
-- Relational
SELECT o.id, o.total, u.name, u.email,
       oi.sku, oi.qty, oi.price
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN order_items oi ON oi.order_id = o.id
WHERE o.user_id = 456
ORDER BY o.created_at DESC;
```

```python
# MongoDB: single query, no joins needed
orders = db.orders.find(
    {"user_id": "usr_456"},
    sort=[("created_at", -1)],
)
# Each order already contains:
#   user_name (denormalized)
#   items[] (embedded)
```

## Relational to DynamoDB

```sql
-- Relational: 3 tables, 2 joins
SELECT * FROM orders WHERE user_id = 456 AND created_at > '2024-01-01';
SELECT * FROM order_items WHERE order_id = 'abc123';
```

```python
# DynamoDB: 2 queries, no joins
# Query 1: user's orders in date range
orders = table.query(
    KeyConditionExpression="PK = :pk AND SK BETWEEN :start AND :end",
    ExpressionAttributeValues={
        ":pk": "USER#usr_456",
        ":start": "ORDER#2024-01-01",
        ":end": "ORDER#2024-12-31",
    },
)

# Query 2: order items (if not embedded)
items = table.query(
    KeyConditionExpression="PK = :pk AND begins_with(SK, :sk)",
    ExpressionAttributeValues={":pk": "ORDER#ord_abc123", ":sk": "ITEM#"},
)
```
