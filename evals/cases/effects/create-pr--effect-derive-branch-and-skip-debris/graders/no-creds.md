---
type: regex
pattern: '^(credentials\.json|scratch-notes\.txt)$'
flags: m
match: not_contains
target: { source: file, path: tracked.txt }
---
