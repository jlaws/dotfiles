---
type: regex
pattern: 'framer\.example\.com'
flags: i
match: not_contains
target: { source: file, path: digest.md }
---
