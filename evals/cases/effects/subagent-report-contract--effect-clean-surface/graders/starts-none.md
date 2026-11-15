---
type: regex
pattern: '^\s*(?:#+\s*)?(?:no\b[^\n]*\b(?:findings?|defects?|issues?|bugs?|problems?)\b|none\b)'
flags: i
target: { source: file, path: report.md }
---
