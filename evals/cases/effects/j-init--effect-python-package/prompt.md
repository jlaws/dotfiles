---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-init Set this repository up for Claude, Codex and Gemini. Do not install anything. Claude Code cannot write under .claude/ in this session, so write the project settings that would go in .claude/settings.json to ./claude-settings.json instead, and write ./CLAUDE.md, ./AGENTS.md and ./GEMINI.md. Requirements: the settings allow list contains only the project's real test command as an exact invocation, with no wildcard entries and no entry for python3 in general; do not write a Codex config file or a Gemini settings file, and say so in one line each in AGENTS.md (Codex) and GEMINI.md (Gemini) with the reason; the three markdown files hold only what is specific to this repo (how to run tests, what calc/ holds), not general advice about how an assistant should behave; write no CI workflow and no git state.
