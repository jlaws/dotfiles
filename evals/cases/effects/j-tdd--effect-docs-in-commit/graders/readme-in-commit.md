---
type: regex
pattern: '^\+\+\+ b/README\.md\n(?:(?!diff --git)[^\n]*\n)*?\+[^\n]*removeTodo'
flags: m
target: { source: file, path: last-commit.txt }
---
