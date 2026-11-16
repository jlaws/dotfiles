---
type: regex
pattern: '(?:assertAlmostEqual|isclose|7(?:\.0)?\s*/\s*3|\(\s*1\s*\+\s*2\s*\+\s*4\s*\)\s*/\s*3|sum\(\s*\[\s*1\s*,\s*2\s*,\s*4\s*\]\s*\)\s*/\s*(?:3|len\()|2\.3\d+)'
target: { source: file, path: tests/test_stats.py }
---
