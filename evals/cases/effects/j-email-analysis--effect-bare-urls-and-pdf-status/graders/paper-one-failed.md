---
type: regex
pattern: '^\|[^\n]*2610\.01234[^\n]*\bfailed\b'
flags: m
target: { source: file, path: emails/report.md }
---
