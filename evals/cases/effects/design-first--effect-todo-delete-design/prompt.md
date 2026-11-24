---
description: "base the design on the existing code, offer 2-3 approaches with a recommendation, and state numbered falsifiable assumptions; .claude/skills/design-first/SKILL.md; hard because addTodo assigns ids from list length, so delete makes a later add reuse an id, visible only by reading the code"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

We want users to delete todos from the list in src/todo.js. No code yet: I want to agree on the shape first. Save the design to ./design.md covering: the behavior, the data flow, error handling, and the testing approach, with two or three approaches and a recommendation, and a numbered list of the assumptions you are making, each stated so it could be proven wrong. Base it on the code that exists. Do not edit src/ or test/.
