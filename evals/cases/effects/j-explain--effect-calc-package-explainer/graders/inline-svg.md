---
type: regex
pattern: '<svg\b[^>]*>[\s\S]*</svg>'
flags: i
target: { source: file, path: calc-explainer.html }
---
