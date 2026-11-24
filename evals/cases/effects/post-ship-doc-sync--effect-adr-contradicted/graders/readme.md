---
type: regex
pattern: '^(?=[^\n]*--input data\.csv)(?=[^\n]*--output data\.jsonl)'
flags: m
target: { source: file, path: README.md }
---
