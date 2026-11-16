---
type: regex
pattern: 'def median\([^)]*\)[^\n]*:(?:(?!\ndef )[\s\S])*?(?:not xs|not s\b|len\((?:xs|s)\)\s*(?:==|<|<=)\s*[01])'
target: { source: file, path: calc/stats.py }
---
