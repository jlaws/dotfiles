---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

We ingest about 200 GB a day of clickstream events and analysts want hourly aggregates in the warehouse. Help me design the pipeline: orchestration, incremental loads, and how to handle late-arriving events.
