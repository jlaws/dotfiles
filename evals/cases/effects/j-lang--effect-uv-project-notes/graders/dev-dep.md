---
type: regex
pattern: 'uv\s+add\s+[^\n]*(?:--dev|--group\s+dev)[^\n]*pytest|uv\s+add\s+[^\n]*pytest[^\n]*(?:--dev|--group\s+dev)'
flags: i
target: { source: file, path: setup-notes.md }
---
