---
type: llm
focus: { source: file, path: src/todo.js }
---

All claims are required.
PASS if removeTodo is exported, returns a new array without the matching todo, and throws an Error when no todo has the given id.
PASS if addTodo no longer derives the new id from list.length alone (for example it uses the highest existing id plus one), so removing a middle todo and then adding does not duplicate an id.
FAIL if removeTodo mutates its input list.
