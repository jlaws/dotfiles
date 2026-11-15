---
type: regex
pattern: '<button\b[\s\S]*(?:clipboard|Blob|download)'
flags: i
target: { source: file, path: drift-report.html }
---
