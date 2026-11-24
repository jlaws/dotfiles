---
type: llm
focus: { source: file, path: src/todo.js }
---

All claims are required.
PASS if src/todo.js exports removeTodo and no longer defines clearCompleted.
FAIL if src/todo.js still defines or exports clearCompleted.
