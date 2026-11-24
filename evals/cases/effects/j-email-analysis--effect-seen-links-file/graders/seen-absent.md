---
type: regex
pattern: '2610\.05678|10k-agent-evals'
match: not_contains
target: { source: file, path: emails/report.md }
---
