---
type: regex
pattern: '^\W*(?:[\w ]*(?:[Mm]ode|[Rr]ecommend\w*|[Vv]erdict)\W*)?(?:SELECTIVE )?EXPAND\b'
target: { source: file, path: scope-review.md }
---
