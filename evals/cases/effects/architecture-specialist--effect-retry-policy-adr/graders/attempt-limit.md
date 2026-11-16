---
type: regex
target: { source: file, path: retry-design.md }
pattern: '(?:attempts?|retries|tries)[^\n]{0,30}?\b\d{1,2}\b|\b\d{1,2}\b[^\n]{0,15}?(?:attempts|retries|tries)\b'
flags: i
---
