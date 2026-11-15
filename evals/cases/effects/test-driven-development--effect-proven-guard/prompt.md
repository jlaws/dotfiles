---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

complete(list, id) in src/todo.js has no tests. Add tests to test/todo.test.js for three behaviors: it marks the item with the given id as done, it leaves the input list unmodified, and it leaves the list unchanged for an id that does not exist.

These behaviors already work, so a new test will pass the first time it runs. A passing test proves nothing yet, so prove each one can fail: for each of the three tests, temporarily break complete() in a way that should turn that test red, run `node --test` and confirm that test fails on its own assertion, then restore src/todo.js and confirm the whole suite passes again.

Save a proof log to ./proof.md with one entry per test: the mutation you applied, the failing assertion you observed, and the output of the final green run. When you finish, src/todo.js must be back to its original behavior.
