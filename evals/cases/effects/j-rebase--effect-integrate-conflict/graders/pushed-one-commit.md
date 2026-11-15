---
type: regex
pattern: '^[0-9a-f]{7,40} [^\n]+\n?$'
target: { source: file, path: pushed.txt }
---
