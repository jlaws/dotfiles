---
type: regex
pattern: 'unsubscribe|digest\.example\.org/view'
flags: i
match: not_contains
target: { source: file, path: emails/report.md }
---
