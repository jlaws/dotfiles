---
type: regex
pattern: '^[^#\n]*\.local\.json'
flags: m
target: { source: file, path: .gitignore }
---
