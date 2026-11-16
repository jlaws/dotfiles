---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:security-reviewer agent to review the code in ./api/ for security vulnerabilities (the agent cannot write files, so save its report to ./security-report.md yourself). For each lead give the state confirmed, UNVERIFIED, or rejected, a severity label of [critical], [high], [medium], or [low], and file:line. This is a single review pass with no independent second reviewer and no running the code against anything, so mark every severity as provisional and do not label a lead confirmed. List the leads you ruled out under rejected with the reason, so I can see they were considered.
