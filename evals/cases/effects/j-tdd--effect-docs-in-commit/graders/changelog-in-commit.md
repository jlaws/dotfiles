---
type: regex
pattern: '^commit: [^\n]*\n(?=(?:(?!commit: )[^\n]*\n)*?\+\+\+ b/CHANGELOG\.md\n(?:(?!diff --git|commit: )[^\n]*\n)*?\+[^\n]*(?:removeTodo|remove[- ]todo))(?=(?:(?!commit: )[^\n]*\n)*?\+\+\+ b/src/todo\.js\n(?:(?!diff --git|commit: )[^\n]*\n)*?\+[^\n]*removeTodo)'
flags: mi
target: { source: file, path: commits.txt }
---
