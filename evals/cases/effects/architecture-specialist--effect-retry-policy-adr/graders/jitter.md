---
type: regex
target: { source: file, path: retry-design.md }
pattern: '\b(?:full|equal|decorrelated)[ -]jitter\b|\bjitter\b[^\n]{0,80}\b(?:random|uniform|rand)|\b(?:random|uniform|rand)[^\n]{0,80}\bjitter\b|\bjitter\b[^\n]{0,80}?(?:\d{1,3} ?%|\bup to\b|\[ ?0 ?,|\b0 ?(?:-|to) ?\d)|\d{1,3} ?%[^\n]{0,40}?\bjitter\b'
flags: i
---
