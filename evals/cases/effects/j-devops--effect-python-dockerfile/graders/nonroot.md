---
type: regex
pattern: '^USER\s+(?!(?:root|0)(?::\S*)?[ \t]*$)\S+'
flags: m
target: { source: file, path: Dockerfile }
---
