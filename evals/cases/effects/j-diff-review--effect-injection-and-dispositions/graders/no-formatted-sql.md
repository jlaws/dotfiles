---
type: regex
pattern: 'SELECT[^\n]*(?:''%s''|["'']\s*%\s*[\w(]|["'']\s*\+\s*\w|\.format\()|\bf["''][^"''\n]*SELECT'
match: not_contains
flags: i
target: { source: file, path: app/users.py }
---
