---
type: regex
pattern: '^openapi:\s*["'']?3\.'
flags: m
target: { source: file, path: openapi.yaml }
---
