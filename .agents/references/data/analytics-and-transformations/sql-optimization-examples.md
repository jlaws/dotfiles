# SQL Optimization Examples

## Index Creation Patterns

```sql
-- Standard B-Tree
CREATE INDEX idx_users_email ON users(email);

-- Composite (order matters - leftmost prefix used)
CREATE INDEX idx_orders_user_status ON orders(user_id, status);

-- Partial (index subset of rows)
CREATE INDEX idx_active_users ON users(email) WHERE status = 'active';

-- Expression
CREATE INDEX idx_users_lower_email ON users(LOWER(email));

-- Covering (index-only scans)
CREATE INDEX idx_users_email_covering ON users(email) INCLUDE (name, created_at);

-- Full-text search
CREATE INDEX idx_posts_search ON posts USING GIN(to_tsvector('english', title || ' ' || body));

-- JSONB
CREATE INDEX idx_metadata ON events USING GIN(metadata);
```

## Cursor-Based Pagination

```sql
-- BAD: OFFSET on large tables
SELECT * FROM users ORDER BY created_at DESC LIMIT 20 OFFSET 100000;

-- GOOD: Cursor-based
SELECT * FROM users
WHERE (created_at, id) < ('2024-01-15 10:30:00', 12345)
ORDER BY created_at DESC, id DESC
LIMIT 20;

-- Requires index
CREATE INDEX idx_users_cursor ON users(created_at DESC, id DESC);
```

## Efficient Aggregation

```sql
-- Approximate count (fast)
SELECT reltuples::bigint AS estimate FROM pg_class WHERE relname = 'orders';

-- Filter before counting
SELECT COUNT(*) FROM orders WHERE created_at > NOW() - INTERVAL '7 days';

-- Filter first, then group
SELECT user_id, COUNT(*) as order_count
FROM orders
WHERE status = 'completed'
GROUP BY user_id
HAVING COUNT(*) > 10;
```

## Batch Operations

```sql
-- Batch insert
INSERT INTO users (name, email) VALUES
    ('Alice', 'alice@example.com'),
    ('Bob', 'bob@example.com'),
    ('Carol', 'carol@example.com');

-- Bulk insert (PostgreSQL)
COPY users (name, email) FROM '/tmp/users.csv' CSV HEADER;

-- Batch update with temp table
CREATE TEMP TABLE temp_user_updates (id INT, new_status VARCHAR);
INSERT INTO temp_user_updates VALUES (1, 'active'), (2, 'active');
UPDATE users u SET status = t.new_status FROM temp_user_updates t WHERE u.id = t.id;
```

## Materialized Views

```sql
CREATE MATERIALIZED VIEW user_order_summary AS
SELECT u.id, u.name, COUNT(o.id) as total_orders,
    SUM(o.total) as total_spent, MAX(o.created_at) as last_order_date
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name;

CREATE INDEX idx_user_summary_spent ON user_order_summary(total_spent DESC);

-- Concurrent refresh (no lock)
REFRESH MATERIALIZED VIEW CONCURRENTLY user_order_summary;
```

## Monitoring Queries

```sql
-- Find slow queries (PostgreSQL)
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;

-- Find missing indexes
SELECT schemaname, tablename, seq_scan, seq_tup_read,
    seq_tup_read / seq_scan AS avg_seq_tup_read
FROM pg_stat_user_tables WHERE seq_scan > 0
ORDER BY seq_tup_read DESC LIMIT 10;

-- Find unused indexes
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

## Maintenance

```sql
ANALYZE users;              -- Update statistics
VACUUM ANALYZE users;       -- Reclaim dead tuples + stats
VACUUM FULL users;          -- Reclaim space (locks table)
REINDEX TABLE users;        -- Rebuild indexes
```
