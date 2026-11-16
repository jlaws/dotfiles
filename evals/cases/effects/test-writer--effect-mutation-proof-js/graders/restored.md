---
type: regex
pattern: '^\s*(?:✔|ok \d+ -)\s*complete does not mutate the input list\b'
flags: m
target: { source: file, path: test-output.txt }
---
