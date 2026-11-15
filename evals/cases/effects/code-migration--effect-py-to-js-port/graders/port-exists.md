---
type: regex
pattern: '^(?=[\s\S]*\bexport\s+(?:(?:async\s+)?function\*?\s*add\b|(?:const|let|var)\s+add\b)|[\s\S]*\bexport\s*\{[^}]*\badd\b)(?=[\s\S]*\bexport\s+(?:(?:async\s+)?function\*?\s*mean\b|(?:const|let|var)\s+mean\b)|[\s\S]*\bexport\s*\{[^}]*\bmean\b)'
target: { source: file, path: js/ops.js }
---
