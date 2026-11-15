---
type: regex
pattern: 'digest\.example\.org/'
flags: i
match: not_contains
target: { source: file, path: emails/report.md }
---
