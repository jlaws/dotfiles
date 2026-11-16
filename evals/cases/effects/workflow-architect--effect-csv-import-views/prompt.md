---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:workflow-architect agent to specify the asynchronous "customer uploads a CSV of up to 50,000 rows to import contacts" workflow before it is built. A background worker processes the file in batches. Two users can upload to the same account at the same time, and the worker can crash mid-file. Save the spec to ./import-spec.md. It must organize the workflow in four views: by workflow step, by component, by user journey, and by state. It must include a state table where every system state has a customer-facing label and an operator-facing view, a concurrent-upload conflict path that never silently overwrites, and a cleanup table giving the lifecycle of the temp file and the partial rows written by a crashed worker.
