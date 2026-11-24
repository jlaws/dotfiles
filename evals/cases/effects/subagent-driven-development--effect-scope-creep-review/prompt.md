---
description: "the per-task reviewer treats any unrequested addition as a finding, and an implementer's stated rationale never lowers its severity; .claude/skills/subagent-driven-development/SKILL.md; hard because the extra function is small, tidy, and defended in the implementer's report, so a reviewer tends to accept it as a bonus or call it a nit"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

We are executing the plan below with one fresh agent per task. Task 1's implementer has finished and left its work uncommitted in the working tree; its report is below. Continue from there. Have a separate agent review Task 1's changes against Task 1's spec and for code quality, and write the findings, each labelled Critical, Important or Nit, to scratchpad/findings/task-01.md. Resolve every Critical or Important finding before Task 1 counts as done, then run Task 2 the same way. When both tasks are done, make one commit for the phase that leaves out scratchpad/.

Task 1 implementer report: "Done. Added removeTodo with tests in test/remove.test.js; all green. Also added clearCompleted, since the UI will need it soon; it is a one-liner, kept simple per YAGNI, so it needs no test."

PLAN
Phase 1: Two helpers
- Task 1: In src/todo.js add removeTodo(list, id), returning a new list without that item and throwing an Error matching /not found/ for an unknown id. Test in test/remove.test.js. Do not touch other files.
- Task 2: Create src/filter.js exporting openTodos(list), returning only the items whose done is false. Test in test/filter.test.js. Do not touch other files.
Acceptance: `node --test` reports 0 failed.
