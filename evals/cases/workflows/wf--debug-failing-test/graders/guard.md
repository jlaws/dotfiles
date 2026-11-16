---
type: regex
pattern: 'def mean\([^)]*\)[^\n]*:[^\n]*\n(?:[ \t]+[^\n]*\n|[ \t]*\n)*?[ \t]+(?:if (?:not xs|not len\(xs\)|len\(xs\)\s*(?:==\s*0|<\s*1|<=\s*0))|return 0\.0|[^\n]*\bif xs else\b)'
target: { source: file, path: calc/ops.py }
---
