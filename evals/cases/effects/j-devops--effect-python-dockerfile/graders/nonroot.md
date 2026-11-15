---
type: regex
pattern: '^USER\s+(?!root\b)\S+'
flags: m
target: { source: file, path: Dockerfile }
---
