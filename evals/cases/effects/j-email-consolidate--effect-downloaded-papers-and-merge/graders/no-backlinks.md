---
type: regex
pattern: 'emails/'
match: not_contains
target: { source: file, path: digest.md }
---
