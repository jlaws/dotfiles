---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "fix instrumentation and alerts: histograms over summaries, no unbounded labels, health checks kept out of the SLI, multi-window burn-rate alerts, runbook links, and no paging on one scrape; .claude/commands/j-observe.md; hard because the seeded summary, per-user labels, and 1-minute error-ratio alert each look reasonable on their own"
---

/kb:j-observe fix the instrumentation and alerting in ./obs/ for our orders API, which has a 99.9% availability SLO. Write the corrected metric definitions to ./obs/metrics.fixed.js and the corrected alert rules to ./obs/alerts.fixed.yml, then save a short rationale to ./obs-review.md. Requirements: latency percentiles must be computable across many instances; no metric label may take unbounded values; health-check traffic must not count toward the SLI; the availability alerts must be built on error-budget burn rate and use both a long and a short window; every alert must carry a runbook link; and a paging alert must not flap on a single bad scrape.
