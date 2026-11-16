---
type: regex
pattern: '^(?:from (?:shop\.text|shop|\.text|\.) import (?:\([^)]*|[^\n]*)\b(?:normalize_name|text)\b|import shop\.text)'
flags: m
target: { source: file, path: shop/orders.py }
---
