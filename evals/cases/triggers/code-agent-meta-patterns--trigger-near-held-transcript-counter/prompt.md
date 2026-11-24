---
description: "a request for a script that counts tool calls in a session transcript, a coding task about agent data rather than agent behavior; .claude/skills/code-agent-meta-patterns/SKILL.md; near: none"
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Write a Python script that reads a Claude Code session transcript in .jsonl format and prints how many times each tool was called.
