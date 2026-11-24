---
type: regex
pattern: '^Verdict: (?:CONCERNS|FAIL)\s*$'
flags: m
target: { source: file, path: review.md }
---
