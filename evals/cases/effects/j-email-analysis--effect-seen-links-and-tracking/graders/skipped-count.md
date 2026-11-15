---
type: regex
pattern: '^[^\n]*Skipped[^\n]*:\s*2\b'
flags: im
target: { source: file, path: emails/report.md }
---
