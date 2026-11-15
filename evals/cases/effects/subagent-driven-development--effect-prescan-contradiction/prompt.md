---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Execute the plan below with one fresh agent per task. Before dispatching anything, read the whole plan and save any contradictions, inconsistent signatures, or ordering problems to ./prescan.md (write "none" if there are none), together with the decision you took for each. Keep function signatures consistent with the existing functions in src/todo.js. Then run the plan with your decisions applied.

PLAN
Phase 1: Removal
- Task 1: In src/todo.js add removeTodo(id, list), returning a new list without that item and throwing an Error matching /not found/ for an unknown id. Test in test/remove.test.js.
- Task 2: In src/todo.js add removeMany(list, ids), which calls removeTodo(list, id) for each id in turn and returns the result. Test in test/remove-many.test.js.
Acceptance: `node --test` reports 0 failed.
