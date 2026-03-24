# DynamoDB Single-Table Design and GSI Patterns

Extended examples for DynamoDB single-table design, GSI overloading, and query patterns.

## Single-Table Design

```python
import boto3
from datetime import datetime

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("app-data")

# All entities in ONE table with overloaded PK/SK

# User entity
table.put_item(Item={
    "PK": "USER#usr_456",
    "SK": "PROFILE",
    "name": "Alice",
    "email": "alice@example.com",
    "created_at": "2024-01-01T00:00:00Z",
    "entity_type": "User",
})

# Order entity (under user partition)
table.put_item(Item={
    "PK": "USER#usr_456",
    "SK": "ORDER#2024-01-15#ord_abc123",  # Date prefix for range queries
    "order_id": "ord_abc123",
    "status": "shipped",
    "total": "44.97",
    "entity_type": "Order",
})

# Order items (under order partition for direct lookup)
table.put_item(Item={
    "PK": "ORDER#ord_abc123",
    "SK": "ITEM#WIDGET-1",
    "sku": "WIDGET-1",
    "name": "Blue Widget",
    "qty": 2,
    "price": "9.99",
    "entity_type": "OrderItem",
})

# Query: Get user profile
resp = table.get_item(Key={"PK": "USER#usr_456", "SK": "PROFILE"})

# Query: Get all orders for user (sorted by date)
resp = table.query(
    KeyConditionExpression="PK = :pk AND begins_with(SK, :sk)",
    ExpressionAttributeValues={":pk": "USER#usr_456", ":sk": "ORDER#"},
    ScanIndexForward=False,  # Newest first
)

# Query: Get orders in date range
resp = table.query(
    KeyConditionExpression="PK = :pk AND SK BETWEEN :start AND :end",
    ExpressionAttributeValues={
        ":pk": "USER#usr_456",
        ":start": "ORDER#2024-01-01",
        ":end": "ORDER#2024-01-31",
    },
)
```

## GSI Overloading

```python
# GSI1: Inverted index (access order by order_id directly)
# Main table: PK=USER#id, SK=ORDER#date#id
# GSI1:       PK=ORDER#id, SK=USER#id
table.put_item(Item={
    "PK": "USER#usr_456",
    "SK": "ORDER#2024-01-15#ord_abc123",
    "GSI1PK": "ORDER#ord_abc123",     # GSI partition key
    "GSI1SK": "USER#usr_456",          # GSI sort key
    "order_id": "ord_abc123",
    "status": "shipped",
    "entity_type": "Order",
})

# Query GSI: Get order by order_id
resp = table.query(
    IndexName="GSI1",
    KeyConditionExpression="GSI1PK = :pk",
    ExpressionAttributeValues={":pk": "ORDER#ord_abc123"},
)

# GSI2: Status index (get all orders by status)
# GSI2PK = STATUS#shipped, GSI2SK = date
```
