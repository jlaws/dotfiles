---
type: regex
pattern: 'END SECTION'
match: not_contains
target: { source: file, path: digest.md }
---
