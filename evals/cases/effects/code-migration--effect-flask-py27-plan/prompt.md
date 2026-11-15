---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Plan the migration of a Python 2.7 Flask service to Python 3.12. Facts: about 400 files and 60k lines; a ledger module with financial rounding rules that has 30% test coverage; user sessions stored as pickles in Redis; a handful of C-extension dependencies with no Python 3 release; the load balancer can send a share of traffic to a second deployment running side by side; there is currently no baseline for latency or error rate.

Save the plan to ./migration-plan.md with: the complexity rating and what drives it, the strategy, the order of the work, the rollback triggers with their thresholds and how rollback is performed, and the timeline.
