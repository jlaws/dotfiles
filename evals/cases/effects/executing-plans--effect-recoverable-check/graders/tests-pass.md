---
type: regex
pattern: '^Ran [0-9]+ tests in [\s\S]*^OK$'
flags: m
target: { source: file, path: tests.txt }
---
