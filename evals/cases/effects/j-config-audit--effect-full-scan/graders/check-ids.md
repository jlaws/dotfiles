---
type: regex
pattern: '^.*(CRITICAL|FAIL|WARN|HIGH).*:\s*\S+:\d+'
flags: mi
target: { source: file, path: config-audit.md }
---
