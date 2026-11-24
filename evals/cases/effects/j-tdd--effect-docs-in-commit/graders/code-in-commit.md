---
type: regex
pattern: '^\+\+\+ b/src/todo\.js\n(?:(?!diff --git|commit: )[^\n]*\n)*?\+[^\n]*removeTodo'
flags: m
target: { source: file, path: commits.txt }
---
