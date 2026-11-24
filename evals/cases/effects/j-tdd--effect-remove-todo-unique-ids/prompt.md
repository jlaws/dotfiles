---
description: "cover happy path, error cases, and edge cases in tests written before the code, and watch them fail before implementing; .claude/commands/j-tdd.md; hard because the id collision after a removal lives in addTodo, which the request does not ask to change, so only an edge-case test exposes it"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-tdd add a removeTodo(list, id) function to src/todo.js. It returns a new list without the todo that has that id and does not mutate the input. It throws an Error when no todo has that id. After a removal, addTodo must never hand out an id that is still in the list. Write the tests first in test/todo.test.js and run them before implementing; save that failing run to ./red-run.txt. Run the full suite with `node --test` when done, then commit with a message that describes the change.
