---
type: regex
target: { source: file, path: retry-design.md }
pattern: '\b(?:full|equal|decorrelated)[ -]jitter\b|\bjitter\b[^\n]{0,80}\b(?:random|uniform|rand)|\b(?:random|uniform|rand)[^\n]{0,80}\bjitter\b'
flags: i
---
