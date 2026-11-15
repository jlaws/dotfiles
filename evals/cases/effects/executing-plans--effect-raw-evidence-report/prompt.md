---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Carry out the plan below. After the last task, run the full suite once more. Then save ./report.md containing the raw terminal output of every verification run, pasted as code blocks and labelled by task (do not paraphrase results), the result of each phase's acceptance check, and a Decision Log of any judgment calls. Commit once per phase.

PLAN
Phase 1: Unknown ids
- Task 1: In test/todo.test.js add a test "complete throws for an unknown id" asserting that complete([], 99) throws an error matching /not found/. Run `node --test`. Expect one failure.
- Task 2: In src/todo.js make complete throw new Error("todo not found") when no item has the given id. Run `node --test`. Expect no failures.
Acceptance: `node --test` reports 0 failed.
Phase 2: Coverage
- Task 3: In test/todo.test.js add a test "complete marks only the matching item done" using a two-item list. Run `node --test`. Expect no failures.
- Task 4: Create README.md documenting addTodo and complete, including the not-found error. No command to run.
Acceptance: `node --test` reports 0 failed.
