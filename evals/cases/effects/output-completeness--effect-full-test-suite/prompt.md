---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "every requested function and test is delivered in full, with no skeleton tests or TODO stubs; .claude/skills/output-completeness/SKILL.md; hard because six functions at two tests each invites stubbing the later ones once the pattern feels implied"
---

Extend src/todo.js with removeTodo(list, id), toggle(list, id), clearCompleted(list), and search(list, text), all pure functions in the same style. Then write real tests in test/todo.test.js for every exported function including the existing complete and the new ones, at least two tests each. Run node --test and make it pass. Save ./SUMMARY.md listing each exported function with the names of its tests. No skeleton tests, no assert.ok(true), no TODO markers.
