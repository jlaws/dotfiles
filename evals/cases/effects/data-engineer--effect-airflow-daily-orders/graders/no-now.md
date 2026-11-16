---
type: regex
pattern: 'datetime\.now\('
match: not_contains
target: { source: file, path: dags/daily_orders_fixed.py }
---
