---
type: regex
pattern: '^(?=[\s\S]*\|[^\n]*\bR1\.1\b)(?=[\s\S]*\|[^\n]*\bR1\.2\b)(?=[\s\S]*\|[^\n]*\bR1\.3\b)(?=[\s\S]*\|[^\n]*\bR2\.1\b)(?=[\s\S]*\|[^\n]*\bR2\.2\b)'
target: { source: file, path: triage.md }
---
