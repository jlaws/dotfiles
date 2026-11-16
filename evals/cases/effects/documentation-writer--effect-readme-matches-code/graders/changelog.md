---
type: regex
pattern: '^##\s*\[?Unreleased\]?[\s\S]*^###\s*Fixed'
flags: m
target: { source: file, path: CHANGELOG.md }
---
