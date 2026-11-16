---
type: regex
pattern: '^#+[^\n]*\bby (?:user journey|journey)\b'
flags: im
target: { source: file, path: import-spec.md }
---
