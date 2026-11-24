---
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "a quick design question about two services; .claude/commands/j-system-design.md; near: j-arch"
---

Quick design question: should service A call service B over plain HTTP or put a queue between them? We have about 20 requests per minute and B is occasionally down for a minute.
