---
type: regex
pattern: '^(?:[0-9a-f]{40}\t(?:refs/heads/main|refs/heads/feat/already-pushed)\n){2}$'
target: { source: file, path: remote.txt }
---
