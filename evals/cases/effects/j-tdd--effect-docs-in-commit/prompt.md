---
description: "before the commit, apply the documentation gate, so a new public function lands with its README and CHANGELOG updates in the same commit; .claude/commands/j-tdd.md; hard because the red-green loop ends naturally at green tests, and nothing in the request mentions docs, so the API list and the Unreleased changelog section stay stale"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-tdd add a removeTodo(list, id) function to src/todo.js and export it. It returns a new list without the todo that has that id and does not mutate the input, and it throws an Error when no todo has that id. Write the tests first in test/todo.test.js. Commit the change in the state you would ship. Then run `node --test > ./tests.txt 2>&1` and `git log -1 -p --format=%s > ./last-commit.txt`.
