---
type: llm
focus: { source: file, path: schema.sql }
---

All claims are required.
PASS if schema.sql defines an index on orders.customer_id and an index on order_items.order_id (a separate CREATE INDEX, not only the primary key).
PASS if email uniqueness is enforced on customers and phone is nullable.
FAIL if any foreign key column lacks its own index, or order_items has no foreign key to orders.
