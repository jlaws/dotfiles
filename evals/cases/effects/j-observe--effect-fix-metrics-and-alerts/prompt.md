---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-observe fix the instrumentation and alerting in ./obs/ for our orders API, which has a 99.9% availability SLO. Write the corrected metric definitions to ./obs/metrics.fixed.js and the corrected alert rules to ./obs/alerts.fixed.yml, then save a short rationale to ./obs-review.md. Requirements: latency percentiles must be computable across many instances; no metric label may take unbounded values; health-check traffic must not count toward the SLI; the availability alerts must be built on error-budget burn rate and use both a long and a short window; every alert must carry a runbook link; and a paging alert must not flap on a single bad scrape.
