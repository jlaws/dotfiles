---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

I want to add removeTodo(list, id) to src/todo.js: it returns a new list without that item and throws an Error whose message matches /not found/ for an unknown id. Plan the work and save the plan to ./plans/plan.md. Write no code.

Every risk the plan names must sit in a Risks table with the file and line, test, or measurement that closes it. Every decision must already be made inside the plan: no TBD, no "choose an appropriate", and no follow-ups or known-limitations section. Name each test together with the assertion it makes, and give each verification command with the output you expect.
