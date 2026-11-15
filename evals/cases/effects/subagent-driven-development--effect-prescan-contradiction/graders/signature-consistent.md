---
type: regex
pattern: '(?:function\s+removeTodo\s*|removeTodo\s*=\s*(?:function\s*)?)\(\s*list\s*,\s*id\b'
target: { source: file, path: src/todo.js }
---
