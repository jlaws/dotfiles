---
type: regex
pattern: '(?:not ok|fail\s+[1-9]|✖)'
flags: i
target: { source: file, path: red-run.txt }
---
