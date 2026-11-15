---
type: regex
pattern: 'def mean.*(?:if not xs|len\(xs\)\s*==\s*0|xs else)'
flags: s
target: { source: file, path: calc/ops.py }
---
