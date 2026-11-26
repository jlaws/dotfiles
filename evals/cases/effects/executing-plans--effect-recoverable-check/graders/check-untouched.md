---
type: regex
pattern: '^LIMIT = 79$[\s\S]*^        if len\(line\) > LIMIT:$'
flags: m
target: { source: file, path: tools/check_style.py }
---
