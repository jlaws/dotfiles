---
type: regex
pattern: '^(?!main$)(?!master$)[a-z0-9][a-z0-9/_.-]*$'
flags: m
target: { source: file, path: branch.txt }
---
