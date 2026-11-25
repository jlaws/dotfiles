---
type: regex
pattern: '^([0-9a-f]{40})\n[\s\S]*\b\1\b'
target: { source: file, path: merges.txt }
---
