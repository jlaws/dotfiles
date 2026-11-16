---
type: regex
pattern: '^\s*(?:✖|not ok \d+ -)\s*complete does not mutate the input list\b'
flags: m
target: { source: file, path: mutant-output.txt }
---
