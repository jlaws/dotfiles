---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "set rollback triggers per signal: error rate over 2x baseline rolls back automatically through the flag, p99 latency over 2x baseline is investigated and rolled back if not fixed in 30 minutes, more than 10 complaints in the first hour disables the feature for new users, and any data corruption rolls back at once with an incident; .claude/references/product/opportunity-and-roadmap.md; hard because the thresholds and the graded actions are house values, and a model tends to give every signal the same full rollback"
---

Use the kb:product-manager agent to set the rollback triggers for the launch of bulk user invite, which ships behind a feature flag to all eligible admins. Cover four signals: the error rate of the invite endpoint, its p99 latency, user complaints, and data integrity problems (for example invited users attached to the wrong organization). For each signal, give the threshold that triggers action, the action taken, and how long to wait before acting (or that the action is immediate). Save the triggers to ./rollback-triggers.md.
