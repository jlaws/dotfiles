---
type: regex
pattern: 'calc/report\.py'
match: not_contains
target: { source: file, path: changed.txt }
---
