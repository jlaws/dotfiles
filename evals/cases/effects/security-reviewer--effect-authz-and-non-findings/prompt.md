---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:security-reviewer agent to review ./invoices/ for authorization flaws and secrets exposure (the agent cannot write files, so save its report to ./security-report.md yourself). Use severity labels [critical], [high], [medium], or [low] with file:line on each finding. Only report issues where a lower-privileged party controls an input that crosses a trust boundary; do not report best-practice deviations, defense-in-depth gaps with no reachable boundary, or effects limited to the person who supplied the input. Put those under a short "Not findings" heading with one line of reasoning each, and keep severity no higher than the damage you can actually trace.
