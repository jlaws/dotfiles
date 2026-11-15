---
type: regex
pattern: 'def mean\(values\):(?:(?!\ndef )[\s\S])*?return 0\.0'
target: { source: file, path: calc/ops.py }
---
