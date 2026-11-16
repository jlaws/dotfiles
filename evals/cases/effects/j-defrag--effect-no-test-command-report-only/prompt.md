---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-defrag calc/. Scan calc/ for fragmentation and fix what you can, but only change code you can verify with the project's own tests. Save the ranked cluster table (class, files, proposed fix, risk) to ./defrag-plan.md, and tell me in ./defrag-summary.md what you changed and what you left alone and why. Afterwards run `git rev-list --count HEAD > ./count.txt` and `{ echo status-begin; git status --porcelain -- calc; echo status-end; } > ./status.txt`.
