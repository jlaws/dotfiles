---
type: regex
pattern: '(?:src|href)\s*=\s*[\"'']\s*(?:https?:)?//|<link\b[^>]*rel\s*=\s*[\"'']?stylesheet|@import'
flags: i
match: not_contains
target: { source: file, path: calc-explainer.html }
---
