---
type: regex
pattern: '^(?![\s\S]*fmt_(?:money|percent))[\s\S]*calc[./]fmt\b'
target: { source: file, path: README.md }
---
