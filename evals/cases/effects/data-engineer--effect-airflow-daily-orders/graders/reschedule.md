---
type: regex
pattern: 'mode\s*=\s*[''"]reschedule[''"]|\bdeferrable\s*=\s*True\b'
target: { source: file, path: dags/daily_orders_fixed.py }
---
