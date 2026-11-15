---
type: llm
focus: { source: file, path: test/todo.test.js }
---

All claims are required.
PASS if createTodo takes an overrides argument that is merged over defaults, and at least two of the new complete() tests build their input with it.
PASS if the new test names each state a behavior and a condition (for example 'marks the item done when the id matches'), not just the function name.
PASS if the new tests cover a matching id, an unknown id, and that the input list is not mutated.
FAIL if any new test reaches into internals or asserts on anything other than the return value of exported functions or the input list.
