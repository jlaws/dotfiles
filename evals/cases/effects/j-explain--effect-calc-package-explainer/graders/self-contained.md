---
type: regex
pattern: '<script\b[^>]*\bsrc\s*=|<link\b|<(?:img|iframe|video|audio|source|embed)\b[^>]*\bsrc\s*=\s*["'']?\s*(?:https?:)?//|@import|url\(\s*["'']?\s*(?:https?:)?//'
flags: i
match: not_contains
target: { source: file, path: calc-explainer.html }
---
