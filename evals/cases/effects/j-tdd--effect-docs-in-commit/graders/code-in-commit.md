---
type: regex
pattern: '^\+\+\+ b/src/todo\.js\n(?:(?!diff --git)[^\n]*\n)*?\+[^\n]*removeTodo'
flags: m
target: { source: file, path: last-commit.txt }
---
