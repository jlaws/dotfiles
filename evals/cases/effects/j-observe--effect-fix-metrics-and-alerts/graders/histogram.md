---
type: regex
pattern: 'new\s+(?:\w+\.)?Histogram\s*\('
target: { source: file, path: obs/metrics.fixed.js }
---
