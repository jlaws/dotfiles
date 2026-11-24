---
type: regex
pattern: '^\s*(?:#+\s*)?No findings\b'
flags: i
target: { source: file, path: report.md }
---
