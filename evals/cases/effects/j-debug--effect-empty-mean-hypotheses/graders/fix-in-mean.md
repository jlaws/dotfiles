---
type: regex
pattern: 'def mean\([^)]*\)[^\n]*:[^\n]*\n(?:[ \t]+[^\n]*\n|[ \t]*\n)*?[ \t]+[^\n]*(?:\bif\b|\btry\b|\bmax\(\s*len\(|\blen\([^)]*\)\s+or\s+1\b)'
target: { source: file, path: calc/ops.py }
---
