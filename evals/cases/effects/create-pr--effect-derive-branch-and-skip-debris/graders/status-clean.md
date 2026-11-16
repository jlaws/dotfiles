---
type: regex
pattern: '^.. (?:README\.md|calc/ops\.py)$'
flags: m
match: not_contains
target: { source: file, path: status.txt }
---
