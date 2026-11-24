---
description: "an XSS question about application route code, not agent configuration; .claude/commands/j-config-audit.md; near: j-audit"
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Our Express app renders user comments with `res.send("<p>" + comment + "</p>")`. Is that an XSS hole, and how do I fix it in the route handler?
