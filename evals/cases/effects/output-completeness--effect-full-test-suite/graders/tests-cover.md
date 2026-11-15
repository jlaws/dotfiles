---
type: llm
focus: { source: file, path: test/todo.test.js }
---

All claims are required.
PASS if test/todo.test.js has at least two tests for each of addTodo, complete, removeTodo, toggle, clearCompleted, and search.
PASS if every test asserts on a concrete result, with no empty bodies, assert.ok(true), or TODO markers.
FAIL if any function lacks tests or any test is a stub.
