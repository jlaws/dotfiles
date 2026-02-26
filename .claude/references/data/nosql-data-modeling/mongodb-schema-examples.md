# MongoDB Schema Design Examples

Extended code examples for MongoDB schema patterns and index strategies.

## Schema Design Patterns

```python
from pymongo import MongoClient
from datetime import datetime

db = MongoClient()["ecommerce"]

# Pattern 1: Embedded (1:few, always read together)
order = {
    "_id": "ord_abc123",
    "user_id": "usr_456",
    "user_name": "Alice",          # Denormalized from users collection
    "created_at": datetime.utcnow(),
    "status": "shipped",
    "items": [                     # Embedded -- always fetched with order
        {"sku": "WIDGET-1", "name": "Blue Widget", "qty": 2, "price": 9.99},
        {"sku": "GADGET-3", "name": "Red Gadget", "qty": 1, "price": 24.99},
    ],
    "total": 44.97,
}
db.orders.insert_one(order)

# Pattern 2: Reference (1:many, unbounded growth)
# Blog post with comments -- comments can grow to thousands
post = {
    "_id": "post_789",
    "title": "NoSQL Modeling",
    "body": "...",
    "comment_count": 0,  # Cached count to avoid counting query
}

comment = {
    "_id": "cmt_001",
    "post_id": "post_789",    # Reference to parent
    "author": "Bob",
    "text": "Great post!",
    "created_at": datetime.utcnow(),
}

# Pattern 3: Bucket pattern (time-series, IoT)
# Instead of one doc per measurement, bucket by hour
sensor_bucket = {
    "_id": "sensor_1_2024010112",  # sensor_id + YYYYMMDDHH
    "sensor_id": "sensor_1",
    "start": datetime(2024, 1, 1, 12, 0),
    "count": 60,
    "measurements": [
        {"ts": datetime(2024, 1, 1, 12, 0, 0), "temp": 22.1, "humidity": 45},
        {"ts": datetime(2024, 1, 1, 12, 1, 0), "temp": 22.3, "humidity": 44},
        # ... up to 60 per hour
    ],
    "avg_temp": 22.2,  # Pre-computed aggregates
    "max_temp": 23.1,
}
```

## Index Strategies

```python
# Compound index for user orders by date (covers patterns 2 and 3)
db.orders.create_index([("user_id", 1), ("created_at", -1)])

# Text index for search
db.posts.create_index([("title", "text"), ("body", "text")])

# TTL index for auto-expiring documents
db.sessions.create_index("expires_at", expireAfterSeconds=0)

# Partial index (only index active orders -- saves space)
db.orders.create_index(
    [("user_id", 1), ("created_at", -1)],
    partialFilterExpression={"status": {"$ne": "cancelled"}},
)
```
