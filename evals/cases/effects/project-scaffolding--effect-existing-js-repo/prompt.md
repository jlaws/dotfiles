---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

This repo has code but no agent configuration. Set it up for Claude Code, Codex, and Gemini CLI: write CLAUDE.md, AGENTS.md, and GEMINI.md at the root, and the project settings file content for .claude/settings.json saved as ./claude-settings.json (files under .claude/ cannot be written here), with permissions for the commands this project routinely needs. Keep permissions minimal and add only what this repo needs. Do not install anything.

Each of the three docs should hold what is true of this repo only. End with ./SCAFFOLD.md listing every file you wrote and every command you ran.
