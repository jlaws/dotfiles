---
type: regex
pattern: '^\s*branch\s*=\s*(?:True|true|1)\b'
flags: m
target: { source: file, path: .coveragerc }
---
