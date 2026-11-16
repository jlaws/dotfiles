---
type: regex
pattern: 'secrets:\s*inherit'
match: not_contains
target: { source: file, path: release.fixed.yml }
---
