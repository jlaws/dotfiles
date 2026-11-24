---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "research what the codebase can answer before deciding, then record each decision and numbered falsifiable assumptions; .claude/commands/j-brainstorm.md; hard because the request names undo in the abstract, and a model can pick a meaning without first reading that calc holds only stateless functions"
---

/kb:j-brainstorm Users keep asking for an undo feature in calc. We do not yet know whether that means undoing the last operation, keeping a history of results, or something else. Nothing should be implemented in this session. Work out what to build and save the design to ./design.md. It must make a decision, with a one-line reason, for each open question rather than leaving questions open; it must have a section titled Out of scope listing at least two things deliberately not built; and it must include a numbered list of assumptions, each stated so it could be proven wrong. Do not create or edit any file other than ./design.md.
