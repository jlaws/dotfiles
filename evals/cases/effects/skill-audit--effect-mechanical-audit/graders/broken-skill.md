---
type: regex
pattern: 'SK-F5[^\n]*broken-skill|broken-skill[^\n]*(?:no|missing)[^\n]*description'
flags: i
target: { source: file, path: audit.md }
---
