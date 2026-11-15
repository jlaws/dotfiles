---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-review-claudemd 2 -- review ./CLAUDE.md against the two session excerpts in ./sessions/. Do not edit CLAUDE.md; write the proposal to ./claudemd-proposal.md with sections headed Violations Found, Missing Patterns, Stale Rules, and Proposed Changes. Each proposed change cites the CLAUDE.md line it touches (as CLAUDE.md:N), says whether to keep, update, or remove it, and gives a one-line reason. Call out any instructions that conflict, any that only restate default behavior, and any that are too specific to be durable.
