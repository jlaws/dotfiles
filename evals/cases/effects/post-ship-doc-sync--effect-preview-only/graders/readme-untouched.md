---
type: regex
pattern: '^(?![\s\S]*README\.md)[\s\S]*\?\? proposed-doc-changes\.diff'
target: { source: file, path: status.txt }
---
