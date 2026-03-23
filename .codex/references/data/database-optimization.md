# Database Optimization

## EXPLAIN ANALYZE Workflow

1. Run `EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)` on the slow query
2. Check for **Seq Scan** on large tables (>10K rows) — likely missing index
3. Compare **estimated vs actual rows** — >10× difference means stale stats → run `ANALYZE tablename`
4. Look for **Nested Loop** on large sets — consider Hash Join via better indexing
5. Check for **Sort** operations without index support — add index matching ORDER BY

### Annotated Example

```sql
EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 42 AND status = 'active';

-- Seq Scan on orders  (cost=0.00..1523.00 rows=5 width=128)
--                      (actual time=12.3..45.6 rows=3 loops=1)
--   Filter: ((user_id = 42) AND (status = 'active'))
--   Rows Removed by Filter: 49997        ← scanning 50K rows to find 3 = needs index
-- Planning Time: 0.1 ms
-- Execution Time: 45.7 ms                ← >10ms OLTP target = slow

-- Fix: CREATE INDEX idx_orders_user_status ON orders(user_id, status);
-- Result: Index Scan → actual time=0.03..0.05 rows=3
```

## N+1 Detection

**Symptom:** N separate queries where 1 would do. Common in ORMs with lazy loading.

```python
# N+1 problem: 1 query for orders + N queries for users
orders = Order.query.all()              # SELECT * FROM orders
for order in orders:
    print(order.user.name)              # SELECT * FROM users WHERE id = ? (×N)

# Fix: eager loading
orders = Order.query.options(joinedload(Order.user)).all()
# SELECT * FROM orders JOIN users ON orders.user_id = users.id (1 query)
```

| Fix Pattern | When to Use | Example |
|-------------|-------------|---------|
| **Eager load (JOIN)** | Related data always needed | `joinedload`, `includes`, `prefetch_related` |
| **Batch load (IN)** | Related data sometimes needed | `WHERE id IN (1,2,3,...N)` |
| **DataLoader** | GraphQL resolvers | Batch + cache per request |

## Connection Pooling

| Mode | Description | Prepared Statements | Use When |
|------|-------------|--------------------:|----------|
| **Transaction** | Connection returned to pool after each transaction | No | Most web apps (short queries) |
| **Session** | Connection held for entire client session | Yes | Long-running connections, prepared statements |
| **Statement** | Connection returned after each statement | No | Simple queries, maximum sharing |

**Pool sizing:** `connections = (cores × 2) + effective_spindle_count` — typically 20–50 for most apps. More connections ≠ more throughput; too many causes contention.

**Warning:** PgBouncer in transaction mode breaks prepared statements and `SET` commands. Use session mode or application-level pooling if you need these.

## Advanced Indexing

| Index Type | Use Case | Example |
|------------|----------|---------|
| **B-tree** (default) | Equality, range, sorting | `CREATE INDEX idx ON t(col)` |
| **Partial** | Subset of rows (reduces size) | `CREATE INDEX idx ON orders(created_at) WHERE status = 'active'` |
| **Covering** | Index-only scans (no heap fetch) | `CREATE INDEX idx ON orders(user_id) INCLUDE (total, status)` |
| **Expression** | Computed values | `CREATE INDEX idx ON users(lower(email))` |
| **GIN** | Full-text, JSONB, arrays | `CREATE INDEX idx ON docs USING gin(to_tsvector('english', body))` |
| **GiST** | Geometric, range, PostGIS | `CREATE INDEX idx ON locations USING gist(coordinates)` |

```sql
-- Partial index: only index active orders (80% smaller if 20% are active)
CREATE INDEX idx_active_orders ON orders(created_at)
WHERE status = 'active';

-- Covering index: avoid heap fetch for common query
CREATE INDEX idx_orders_lookup ON orders(user_id)
INCLUDE (status, total, created_at);
```

## Zero-Downtime Migration Patterns

| Pattern | Risk | Mitigation |
|---------|------|------------|
| **CREATE INDEX CONCURRENTLY** | Slower build, can fail if errors | Don't run in transaction; retry on failure |
| **Add column with default** | PG11+ is instant; older locks table | Check PG version; backfill separately on old versions |
| **Backfill large table** | Long-running UPDATE locks rows | Batch: `UPDATE ... WHERE id BETWEEN x AND y LIMIT 1000` in loop |
| **Rename column** | Breaks reads/writes during transition | Add new → dual-write → migrate reads → drop old |
| **Drop column** | Irreversible | Mark unused first; drop after 1 deploy cycle confirms no reads |
| **Change column type** | May rewrite entire table | Add new column → backfill → swap → drop old |

## Query Performance Targets

| Percentile | OLTP Target | Action If Exceeded |
|------------|-------------|-------------------|
| p50 | <10ms | — |
| p95 | <50ms | Investigate; likely needs index |
| p99 | <200ms | Optimize or add to slow query log |
| Any query | >1s | Mandatory optimization or rewrite |

**Monitoring:** Log all queries >100ms. Review weekly. Track query count per endpoint to catch N+1 regressions.
