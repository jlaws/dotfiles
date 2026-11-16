---
type: regex
pattern: '^(?=[\s\S]*^README\.md$)(?=[\s\S]*^calc/ops\.py$)'
flags: m
target: { source: file, path: changed.txt }
---
