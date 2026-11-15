---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

src/todo.js has `complete(list, id)` with no tests. Add a test suite for it to test/todo.test.js, keeping the existing tests, and run `node --test` to confirm everything passes.

Requirements: build every todo used in the new tests through one `createTodo(overrides)` helper defined in test/todo.test.js, so no test repeats a hand-written todo literal. Name each new test as a sentence stating the behavior and the condition. Test only through the exported functions. Then save ./test-notes.md listing what you deliberately chose not to test and why.
