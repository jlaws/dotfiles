---
type: regex
pattern: '^[0-9a-f]{4,} (?!Initial commit\n)[^\n]+\n[0-9a-f]{4,} Initial commit\n?$'
target: { source: file, path: log.txt }
---
