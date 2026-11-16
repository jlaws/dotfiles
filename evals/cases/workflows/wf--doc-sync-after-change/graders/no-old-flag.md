---
type: regex
pattern: '--out(?!put)'
match: not_contains
target: { source: file, path: README.md }
---
