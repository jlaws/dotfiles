---
type: regex
pattern: '"@types/react"\s*:\s*"[~^]?17'
match: not_contains
target: { source: file, path: package.json }
---
