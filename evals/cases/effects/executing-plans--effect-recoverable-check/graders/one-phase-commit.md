---
type: regex
pattern: '^[0-9a-f]{7,} (?!Initial commit).+\n[0-9a-f]{7,} Initial commit\n?$'
target: { source: file, path: log.txt }
---
