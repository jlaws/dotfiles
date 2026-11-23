---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

We're moving search from SQL LIKE queries to Elasticsearch, touching the indexer job, the query API, and the deploy scripts. Draft the ordered implementation plan, with files and checks per step, before we start.
