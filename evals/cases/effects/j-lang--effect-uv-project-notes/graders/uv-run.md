---
type: regex
pattern: 'uv\s+run\s+(?:python\s+-m\s+)?(?:--\S+\s+)*pytest|uv\s+run\s+python\s+-m\s+pytest'
flags: i
target: { source: file, path: setup-notes.md }
---
