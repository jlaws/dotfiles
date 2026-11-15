---
type: regex
pattern: 'def mean\(xs\):[^\n]*\n(?:[^\n]*\n)*?[^\n]*(?:\bif\b|\btry\b)'
target: { source: file, path: calc/ops.py }
---
