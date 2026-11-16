---
type: regex
pattern: '^[0-9a-f]{40}\trefs/heads/(?!main$|master$)\S+$'
flags: m
target: { source: file, path: remote.txt }
---
