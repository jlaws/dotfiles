---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "with 10 to 25 percent of the error budget left, freeze features so only reliability and bug-fix work ships; .claude/references/devops/sre-practices.md; hard because about a fifth of the budget left reads as a caution zone where a small feature or a planned migration can still go out"
---

Use the kb:devops-engineer agent to decide which of four changes our team may deploy to the orders API for the rest of the current SLO window. The SLO is 99.9% availability over a rolling 30 days, and so far in this window the API has been down for 35 minutes in total. The changes: (1) a new endpoint behind a "reorder" button; (2) an upgrade of the orders PostgreSQL database from version 14 to 16; (3) a fix for a bug that rounds order totals wrongly when a discount applies; (4) timeouts and retries with backoff on the payment client, added after last week's outage. Save ./budget-decision.md with the error budget for the window in minutes, how much of it remains, the deploy policy that remaining budget puts the team under, and a ship or hold call for each change with the reason.
