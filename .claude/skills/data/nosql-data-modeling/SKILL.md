---
name: nosql-data-modeling
description: "Use when designing MongoDB, DynamoDB, or Redis data models, implementing access-pattern-driven design, or migrating from relational to NoSQL databases."
---

# NoSQL Data Modeling

## Database Selection Table

| Factor | MongoDB | DynamoDB | Redis | Firestore |
|--------|---------|----------|-------|-----------|
| Data model | Document (JSON) | Key-value + document | Key-value + data structures | Document (nested) |
| Query flexibility | High (ad-hoc queries) | Low (key-based only) | Low (key-based) | Medium (indexed fields) |
| Scale model | Sharded clusters | Fully managed, infinite | Single-node or cluster | Fully managed |
| Consistency | Tunable (strong or eventual) | Tunable per-request | Strong (single node) | Strong within entity group |
| Cost model | Self-host or Atlas | Pay per RCU/WCU | Memory-based | Pay per read/write ops |
| Best for | General purpose, flexible schemas | Predictable high-scale workloads | Caching, sessions, leaderboards | Mobile/web apps, real-time sync |
| Avoid when | Need ACID joins | Need ad-hoc queries | Data > memory | Need complex queries |
| Max item size | 16 MB | 400 KB | 512 MB (value) | 1 MB |

## Access-Pattern-Driven Design

NoSQL design is backwards from relational. Start with queries, not entities.

### Step-by-Step Process

```
1. List ALL access patterns (reads and writes)
2. Estimate frequency and latency requirements per pattern
3. Choose primary key to satisfy the most critical patterns
4. Design secondary indexes for remaining patterns
5. Denormalize data to avoid joins
6. Accept data duplication as a tradeoff for read performance
```

### Example: E-Commerce

```
Access Patterns:
  1. Get order by order_id                    (100k/day, <10ms)
  2. Get all orders for a user                (50k/day, <50ms)
  3. Get all orders in date range for a user  (10k/day, <100ms)
  4. Get order items for an order             (100k/day, <10ms)
  5. Get user profile                         (200k/day, <10ms)

Design decisions:
  - Pattern 1,4: embed items IN the order document (no join needed)
  - Pattern 2,3: user_id as partition key, order_date as sort key
  - Pattern 5: separate user collection/table
  - Denormalize: store user_name in order (avoid lookup for display)
```

## MongoDB Patterns

### Embed vs Reference Decision

| Factor | Embed | Reference |
|--------|-------|-----------|
| Read together? | Always | Sometimes |
| Array growth | Bounded (<100) | Unbounded |
| Update frequency | Low | High (independent updates) |
| Document size | Fits in 16MB | Would exceed limit |
| Data duplication OK? | Yes | No (single source of truth) |

For schema design code and index strategies, see [references/mongodb-schema-examples.md](references/mongodb-schema-examples.md).

## DynamoDB Patterns

### Partition Key Selection

| Pattern | Key Design | Rationale |
|---------|-----------|-----------|
| User data | `USER#{user_id}` | Natural partition, bounded size |
| Time-series | `SENSOR#{id}#YYYY-MM-DD` | Prevent hot partition; shard by day |
| High-write | `ITEM#{id}#SHARD#{0-9}` | Write sharding for hot keys |
| Global config | `CONFIG#GLOBAL` | Single item, cache it |

For single-table design examples and GSI overloading patterns, see [references/single-table-patterns.md](references/single-table-patterns.md).

## Redis Data Structures

| Structure | Use When | Example |
|-----------|---------|---------|
| String | Simple key-value, counters, cache | Session data, feature flags |
| Hash | Object with fields | User profile fields |
| List | Ordered collection, queue | Job queue, recent items |
| Set | Unique members, intersections | Tags, online users |
| Sorted Set | Ranked data, range queries | Leaderboards, rate limiting |
| Stream | Event log, pub/sub with history | Activity feed, event sourcing |

For Redis code examples (hashes, sorted sets, rate limiting, streams), see [references/redis-patterns.md](references/redis-patterns.md).

## Consistency Patterns

| Pattern | Consistency | Use When |
|---------|------------|----------|
| Read-your-writes | Session-level | User sees their own updates immediately |
| Eventual consistency | None guaranteed | Analytics, feeds, non-critical reads |
| Strong consistency | Immediate | Financial data, inventory counts |
| Write-behind cache | Eventual | High-read, tolerate stale |

```python
# DynamoDB: strong consistency per-read
resp = table.get_item(
    Key={"PK": "USER#456", "SK": "BALANCE"},
    ConsistentRead=True,  # Costs 2x RCU but guarantees latest
)

# MongoDB: read concern + write concern
from pymongo import ReadPreference, WriteConcern

# Strong: write to majority, read from primary
collection = db.get_collection(
    "orders",
    write_concern=WriteConcern(w="majority"),
    read_preference=ReadPreference.PRIMARY,
)

# Eventual: read from secondaries (lower latency, possibly stale)
collection_eventual = db.get_collection(
    "orders",
    read_preference=ReadPreference.SECONDARY_PREFERRED,
)
```

## Migration from Relational

### Step-by-Step

```
1. Map access patterns (not tables)
   - List every SQL query your app runs
   - Group by frequency and latency requirement

2. Denormalize JOIN results
   - If you always JOIN orders + users: embed user_name in order
   - If you sometimes JOIN: reference with user_id

3. Handle relationships
   - 1:1 -> embed
   - 1:few (bounded) -> embed array
   - 1:many (unbounded) -> reference (separate collection/item)
   - many:many -> reference array on one side, or adjacency list

4. Replace transactions
   - Single-document operations are atomic in MongoDB
   - Use DynamoDB TransactWriteItems for multi-item
   - Redesign to minimize multi-document transactions

5. Migrate incrementally
   - Dual-write to both databases during transition
   - Shadow-read from NoSQL, compare with SQL results
   - Switch reads to NoSQL once validated
   - Remove SQL writes last
```

For side-by-side SQL-to-NoSQL query comparisons, see [references/migration-examples.md](references/migration-examples.md).

## Gotchas

- **Modeling entities before access patterns**: NoSQL design starts with queries, not ER diagrams; design for reads, not normalization
- **Unbounded arrays in MongoDB**: embedding 10k comments in a post hits the 16MB limit; reference instead and paginate
- **Hot partitions in DynamoDB**: a single PK receiving disproportionate traffic throttles; add write sharding for hot keys
- **DynamoDB 400KB item limit**: embed carefully; large items hit the limit fast; store blobs in S3, reference by key
- **Scanning instead of querying**: DynamoDB full table scans are expensive and slow; if you need one, your key design is wrong
- **Redis as primary database**: Redis is a cache/data-structure server; data loss on restart unless using AOF persistence; always have a source of truth elsewhere
- **Ignoring GSI costs in DynamoDB**: every GSI duplicates data and consumes its own capacity; 5 GSIs on a hot table = 6x write cost
- **MongoDB without indexes**: queries without index support cause collection scans; use `explain()` to verify index usage
- **Eventual consistency surprises**: write then immediately read may return stale data; use strong consistency for read-after-write patterns
- **Over-denormalization**: duplicating user email in 10 collections means updating 10 places when it changes; denormalize what's read-heavy and rarely updated
- **Forgetting TTL**: cache entries and session data without expiry grow forever; set TTLs on everything temporal
