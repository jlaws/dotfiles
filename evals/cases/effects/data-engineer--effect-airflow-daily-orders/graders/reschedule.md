---
type: regex
pattern: 'mode\s*=\s*[''"]reschedule[''"]'
target: { source: file, path: dags/daily_orders_fixed.py }
---
