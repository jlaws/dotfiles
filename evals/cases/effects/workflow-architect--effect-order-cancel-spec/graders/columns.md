---
type: regex
pattern: '^\|[^\n]*Resource[^\n]*Created By[^\n]*Destroyed By[^\n]*Cleanup Method[^\n]*Timeout'
flags: im
target: { source: file, path: cancel-spec.md }
---
