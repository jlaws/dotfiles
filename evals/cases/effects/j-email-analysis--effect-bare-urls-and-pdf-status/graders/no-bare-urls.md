---
type: regex
pattern: '(?<!\]\()(?<!<)https?://(?![^\[\]\n]*\]\()'
match: not_contains
target: { source: file, path: emails/report.md }
---
