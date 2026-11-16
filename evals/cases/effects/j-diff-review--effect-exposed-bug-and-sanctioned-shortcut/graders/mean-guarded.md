---
type: regex
pattern: 'def mean\([^)]*\)[^\n]*:(?:(?!\ndef )[\s\S])*?(?:not xs|len\(xs\)\s*(?:==|<|<=)\s*[01]|if xs|try:)'
target: { source: file, path: calc/ops.py }
---
