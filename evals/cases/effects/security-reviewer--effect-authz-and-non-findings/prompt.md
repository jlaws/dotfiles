---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "report only issues where a lower-privileged party crosses a trust boundary, here the missing ownership check, and list self-impact and untraced settings under Not findings; .claude/agents/security-reviewer.md; hard because the local CLI file read looks like path traversal and the permissive settings look like findings"
---

Use the kb:security-reviewer agent to review ./invoices/ for authorization flaws and secrets exposure (the agent cannot write files, so save its report to ./security-report.md yourself). Use severity labels [critical], [high], [medium], or [low] with file:line on each finding. Only report issues where a lower-privileged party controls an input that crosses a trust boundary; do not report best-practice deviations, defense-in-depth gaps with no reachable boundary, or effects limited to the person who supplied the input. Put those under a short "Not findings" heading with one line of reasoning each. The settings.py values count as findings only if you can trace an attacker path through the code in ./invoices/ to real damage; otherwise list them under "Not findings". Keep severity no higher than the damage you can actually trace.
