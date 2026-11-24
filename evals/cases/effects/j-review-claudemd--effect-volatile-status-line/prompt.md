---
description: "content that changes week to week, such as counts and dates, is removed from CLAUDE.md rather than refreshed, while non-obvious project conventions stay; .claude/commands/j-review-claudemd.md; hard because a session hands over fresh sprint and bug numbers, so the obvious proposal is to update the status line instead of removing it"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-review-claudemd 2 -- review ./CLAUDE.md against the two session excerpts in ./sessions/. Do not edit CLAUDE.md; write the proposal to ./claudemd-proposal.md with sections headed Violations Found, Missing Patterns, Stale Rules, and Proposed Changes. Each proposed change cites the CLAUDE.md line it touches (as CLAUDE.md:N) and the session it comes from, says whether to keep, update, or remove that line, and gives a one-line reason.
