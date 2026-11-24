---
description: "register a new command with its Codex and Gemini counterparts plus the shared cmd-j skill, in the layout the repo already uses; .claude/commands/j-new.md; hard because the obvious deliverable is the one command file, and the shared cmd-j-demo skill under .agents/ reads as a separate skill rather than part of a command"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-new command j-standup. It summarizes the last day of git commits into a short standup note and takes an optional author argument. Claude Code cannot write under .claude/ in this session, so write the command itself to ./draft/commands/j-standup.md, and write every other file the new command needs in its real place. The command must be usable from every tool this repository configures, not only Claude Code. When you are done, run `find . -path '*standup*' -type f | sort > ./created.txt`.
