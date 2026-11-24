---
description: "delegate the code review to security-reviewer and verify its findings against the code before presenting, each with a STRIDE category and an impact x likelihood score; .claude/commands/j-audit.md; hard because get_profile runs SQL too but binds its parameter, and an unverified pattern scan reports it with the real injections"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-audit audit the login module in ./app.py, covering every function in the file, not only login. Save the findings to ./audit.md. Start with a short threat model of the login flow, then list findings. For every finding give file:line, a severity of Critical, High, Medium, or Low, the STRIDE category it falls under, and a risk score written as impact x likelihood with each rated 1 to 4 (for example "4 x 3 = 12"). Do not report code that is already safe.
