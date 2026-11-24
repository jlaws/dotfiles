---
type: regex
pattern: '^commit: [^\n]*\n(?=(?:(?!commit: )[^\n]*\n)*?\+\+\+ b/README\.md\n(?:(?!diff --git|commit: )[^\n]*\n)*?\+[^\n]*removeTodo)(?=(?:(?!commit: )[^\n]*\n)*?\+\+\+ b/src/todo\.js\n(?:(?!diff --git|commit: )[^\n]*\n)*?\+[^\n]*removeTodo)'
flags: m
target: { source: file, path: commits.txt }
---
