---
type: regex
pattern: 'https?://'
match: not_contains
target: { source: file, path: rebuttal.md }
---
