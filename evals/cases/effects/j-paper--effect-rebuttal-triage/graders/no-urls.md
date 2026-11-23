---
type: regex
pattern: '(?:[Hh][Tt][Tt][Pp][Ss]?://|www\.|doi\.org/|arxiv\.org/)'
match: not_contains
target: { source: file, path: rebuttal.md }
---
