---
type: regex
pattern: '^status-begin\nstatus-end$'
flags: m
target: { source: file, path: status.txt }
---
