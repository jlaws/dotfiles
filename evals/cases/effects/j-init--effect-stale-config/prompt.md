---
description: "for partial config, strip restated global guidance, narrow over-broad allow entries read as untrusted input, and record what changed and why; .claude/commands/j-init.md; hard because the existing wildcard grants and working-style lines look like choices to preserve out of deference"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-init This repository's agent config is stale and partial. Claude Code cannot write under .claude/ in this session, so the existing project settings are in ./claude-settings.json (they stand in for .claude/settings.json); update that file and ./CLAUDE.md in place, and add ./AGENTS.md and ./GEMINI.md. Do not install anything. Requirements: CLAUDE.md keeps only repo-specific facts and states the exact test command; the generic working-style advice already in it is removed because the user's global config already supplies it; the settings allow list is narrowed to the project's test command as an exact invocation, with no wildcard entries; write ./init-notes.md saying which existing entries you removed or narrowed and why, and flag anything over-broad you found.
