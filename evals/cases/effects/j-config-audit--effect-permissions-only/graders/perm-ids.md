---
type: regex
pattern: '^[^\n]*\[[A-Za-z]+\][*_]*\s*`?[A-Za-z][\w.-]*`?[*_]*\s*:\s*`?[^\s:`]+:\d+'
flags: m
target: { source: file, path: permissions-audit.md }
---
