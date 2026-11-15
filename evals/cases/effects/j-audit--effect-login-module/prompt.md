---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-audit audit the login module in ./app.py. Save the findings to ./audit.md. Start with a short threat model of the login flow, then list findings. For every finding give file:line, a severity of Critical, High, Medium, or Low, the STRIDE category it falls under, and a risk score written as impact x likelihood with each rated 1 to 4 (for example "4 x 3 = 12"). Do not report code that is already safe.
