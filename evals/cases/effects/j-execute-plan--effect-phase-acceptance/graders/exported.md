---
type: regex
pattern: '^[^#\n]*\bimport\s+(?:\([^)]*\bmedian\b|[^#\n]*(?:\bmedian\b|\*))|^[^#\n]*\b__all__\b[^\])]*\bmedian\b'
flags: m
target: { source: file, path: calc/__init__.py }
---
