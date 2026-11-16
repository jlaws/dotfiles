---
type: regex
target: { source: file, path: branch.txt }
pattern: '^(?!main$)[a-z]+/[a-z0-9-]+$'
flags: m
---
