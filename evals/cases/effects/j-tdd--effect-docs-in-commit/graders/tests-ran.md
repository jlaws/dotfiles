---
type: regex
pattern: '^(?:# |ℹ )pass [1-9][0-9]*[ \t]*$'
flags: m
target: { source: file, path: tests.txt }
---
