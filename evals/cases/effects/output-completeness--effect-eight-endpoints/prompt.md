---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Write ./api.md documenting these eight endpoints of our inventory API: GET /items, POST /items, GET /items/{id}, PUT /items/{id}, DELETE /items/{id}, POST /items/{id}/reserve, POST /items/{id}/release, GET /items/low-stock. Every endpoint gets its own section with a method and path heading, a parameters table, an example request, an example response, and its error codes. Every section must stand on its own: no "same as above", "similar to", or "see previous" shortcuts, and no placeholder text.
