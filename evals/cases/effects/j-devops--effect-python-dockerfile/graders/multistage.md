---
type: regex
pattern: '^FROM\b[\s\S]*^FROM\b'
flags: m
target: { source: file, path: Dockerfile }
---
