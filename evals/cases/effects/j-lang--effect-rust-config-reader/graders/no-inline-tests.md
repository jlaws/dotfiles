---
type: regex
pattern: '#\s*\[\s*test\s*\]'
match: not_contains
target: { source: file, path: main.rs }
---
