---
type: regex
pattern: '<script\b[^>]*\bsrc\s*=|<link\b|<(?:img|iframe|video|audio|source|embed)\b[^>]*\bsrc\s*=\s*["'']?\s*(?:https?:)?//|@import|url\(\s*["'']?\s*(?:https?:)?//'
flags: i
target: { source: file, path: drift-report.html }
---
