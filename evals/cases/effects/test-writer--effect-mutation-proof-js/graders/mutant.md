---
type: regex
pattern: '^\S*\s*fail [1-9]'
flags: m
target: { source: file, path: mutant-output.txt }
---
