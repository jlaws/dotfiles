---
type: regex
pattern: 'releases/tag/v0\.4\.0'
match: not_contains
target: { source: file, path: emails/report.md }
---
