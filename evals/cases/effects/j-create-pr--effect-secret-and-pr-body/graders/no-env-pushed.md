---
type: regex
pattern: '^\.env$'
flags: m
match: not_contains
target: { source: file, path: pushed.txt }
---
