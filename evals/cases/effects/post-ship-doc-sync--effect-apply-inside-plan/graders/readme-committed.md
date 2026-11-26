---
type: regex
pattern: '^(?=[\s\S]*\?\? log\.txt)(?![\s\S]*README\.md)'
target: { source: file, path: status.txt }
---
