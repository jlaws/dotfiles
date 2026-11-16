---
type: regex
pattern: '(?:^[0-9a-f]{7,40} [^\n]+\n){3}^[0-9a-f]{7,40} [^\n]+'
flags: m
target: { source: file, path: log.txt }
---
