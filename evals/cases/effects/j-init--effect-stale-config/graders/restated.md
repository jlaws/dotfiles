---
type: regex
target: { source: file, path: CLAUDE.md }
pattern: 'Never claim success|Keep answers brief|clarifying questions'
flags: i
match: not_contains
---
