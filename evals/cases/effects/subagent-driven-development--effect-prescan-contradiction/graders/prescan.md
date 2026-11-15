---
type: llm
focus: { source: file, path: prescan.md }
---

All claims are required.
PASS if prescan.md points out that Task 1 defines removeTodo(id, list) while Task 2 calls removeTodo(list, id), an argument-order contradiction.
PASS if prescan.md records a decision for it (for example list first, matching addTodo and complete in src/todo.js).
FAIL if prescan.md says none or only reports unrelated issues.
