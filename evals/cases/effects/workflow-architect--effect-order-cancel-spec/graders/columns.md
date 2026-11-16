---
type: regex
pattern: '^\|[^\n]*Resource[^\n]*Created By[^\n]*Destroyed By[^\n]*Cleanup Method[^\n]*Timeout'
flags: m
target: { source: file, path: cancel-spec.md }
---
