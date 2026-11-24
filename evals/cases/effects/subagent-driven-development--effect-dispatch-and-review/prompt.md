---
description: "each task goes to a fresh agent through a brief file, a separate reviewer returns spec compliance then code quality with severity labels, and the phase commits once; .claude/skills/subagent-driven-development/SKILL.md; hard because a model tends to commit per task or fold the review into the implementer"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Execute the plan below by giving each task to its own fresh agent. Write each task's requirements to scratchpad/briefs/task-01.md and scratchpad/briefs/task-02.md and hand agents the file path, not the plan text. Implementing agents must not commit. Have a separate agent review each task and write its findings to scratchpad/findings/task-01.md and scratchpad/findings/task-02.md with two labelled verdicts, spec compliance first and then code quality, with each finding labelled Critical, Important or Nit. When both tasks pass, make one commit for the whole phase that does not include anything under scratchpad. Save the plan to ./plans/plan.md and record in its Decision Log which execution mode you chose and what decided it. Finish with `git log --oneline > ./log.txt`.

PLAN
Phase 1: Two helpers
- Task 1: In src/todo.js add removeTodo(list, id), returning a new list without that item and throwing an Error matching /not found/ for an unknown id. Test in test/remove.test.js. Do not touch other files.
- Task 2: Create src/filter.js exporting openTodos(list), returning only the items whose done is false. Test in test/filter.test.js. Do not touch other files.
Acceptance: `node --test` reports 0 failed.
