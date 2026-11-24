---
type: regex
pattern: '^(?=[^\n]*needs manual download)[^\n]*\b403\b'
flags: mi
target: { source: file, path: digest.md }
---
