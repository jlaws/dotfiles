---
type: regex
pattern: 'def subtract\([^)]*\)[^\n]*:(?:(?!\ndef )[\s\S])*?raise ValueError'
target: { source: file, path: calc/ops.py }
---
