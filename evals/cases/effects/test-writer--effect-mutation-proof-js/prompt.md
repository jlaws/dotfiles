---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:test-writer agent to add tests for `complete` in src/todo.js. Put them in test/complete.test.js using node --test, and cover: it marks the matching item done, it leaves other items untouched, it does not mutate the list or items passed in (freeze the input), and an unknown id returns an equal list. Name the no-mutation test exactly 'complete does not mutate the input list'. Do not change the code in src/todo.js in the end. To prove the no-mutation test can fail, temporarily change `complete` to set `item.done = true` in place and return the same list, run `node --test` and save the full output to ./mutant-output.txt (the no-mutation test must fail there), then restore src/todo.js exactly as it was and save the full output of the passing run, with the no-mutation test passing again, to ./test-output.txt.
