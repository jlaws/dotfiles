---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-review-claudemd 3 -- review ./CLAUDE.md against the three session excerpts in ./sessions/. Do not edit CLAUDE.md; write the proposal to ./claudemd-proposal.md with sections headed Violations Found, Missing Patterns, Stale Rules, and Proposed Changes. Each proposed change cites the CLAUDE.md line it touches (as CLAUDE.md:N) and the session it comes from. Only propose rules for durable patterns, not one-off preferences.
