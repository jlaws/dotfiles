---
type: regex
pattern: '^### Podcasts[^\n]*\n[\s\S]*^### Funding[^\n]*\n[\s\S]*^### Conferences[^\n]*\n[\s\S]*^### Cultural'
flags: m
target: { source: file, path: digest.md }
---
