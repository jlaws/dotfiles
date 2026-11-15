---
type: regex
pattern: 'def subtract\([^)]*\):(?:(?!\ndef )[\s\S])*?raise ValueError'
target: { source: file, path: calc/ops.py }
---
