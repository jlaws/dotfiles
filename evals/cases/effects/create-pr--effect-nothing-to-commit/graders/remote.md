---
type: regex
pattern: '^(?:[0-9a-f]{40}\t(?:HEAD|refs/heads/main|refs/heads/feat/already-pushed)\n){3}$'
target: { source: file, path: remote.txt }
---
