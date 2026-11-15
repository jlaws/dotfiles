---
type: regex
pattern: '(?=[\s\S]*--output\b)[\s\S]*--out\b(?!put)'
target: { source: file, path: findings/readme.md }
---
