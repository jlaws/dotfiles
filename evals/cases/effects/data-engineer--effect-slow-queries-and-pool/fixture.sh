set -euo pipefail
cat > schema.sql <<'SQL'
-- PostgreSQL 15. Row counts: users 5 million, orders 80 million.
-- The only indexes are the two primary keys.
CREATE TABLE users (
    id         bigint PRIMARY KEY,
    email      text NOT NULL,
    plan       text,
    created_at timestamptz NOT NULL
);

CREATE TABLE orders (
    id         bigint PRIMARY KEY,
    user_id    bigint NOT NULL REFERENCES users (id),
    status     text NOT NULL,
    total      numeric(12,2) NOT NULL,
    created_at timestamptz NOT NULL
);
SQL
cat > slow.sql <<'SQL'
-- Q1: daily paid-orders report
SELECT id, user_id, total FROM orders
WHERE date(created_at) = '2026-11-01' AND status = 'paid';

-- Q2: login lookup
SELECT id, plan FROM users WHERE lower(email) = 'ada@example.com';

-- Q3: order history page 2001 (50 per page), newest first
SELECT id, total, status FROM orders
WHERE user_id = 42
ORDER BY created_at DESC
OFFSET 100000 LIMIT 50;
SQL
