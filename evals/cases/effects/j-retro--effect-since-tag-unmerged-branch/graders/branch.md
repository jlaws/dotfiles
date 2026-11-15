---
type: regex
pattern: '^(?=[^\n]*feat/median)[^\n]*(?:\bunmerged\b|\bnot\s+(?:yet\s+)?merged\b|\b2\b)'
flags: im
target: { source: file, path: retro-since-v1.md }
---
