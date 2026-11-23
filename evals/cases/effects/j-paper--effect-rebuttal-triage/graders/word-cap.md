---
type: regex
pattern: '(?:\S+\s+){400}\S'
match: not_contains
target: { source: file, path: rebuttal.md }
---
