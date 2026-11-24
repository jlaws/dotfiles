---
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "React effect cleanup for a websocket subscription; .claude/commands/j-lang.md; near: j-frontend"
---

In my React component a useEffect subscribes to a websocket and I think it leaks after unmount. How should I write the cleanup correctly?
