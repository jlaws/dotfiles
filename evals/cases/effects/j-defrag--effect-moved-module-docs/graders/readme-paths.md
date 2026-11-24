---
type: regex
pattern: '^(?![\s\S]*fmt_(?:money|percent))[\s\S]*(?:calc[./]fmt\b|\bfrom\s+calc\s+import\s+(?:\(\s*)?(?:\w+\s*,\s*)*fmt\b)'
target: { source: file, path: README.md }
---
