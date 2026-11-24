---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "inventory what already exists before specifying a workflow, including cron jobs and database triggers in migration files, so the spec covers the paths outside the request handler; .claude/references/architecture/workflow-specification.md; hard because the visible cancel path ends at a status update and an email, while the provider cancellation runs from a SQL trigger and a poller, and a nightly renewal job can charge mid-cancel"
---

Use the kb:workflow-architect agent to specify the existing "customer cancels a subscription" workflow in this repository before we change it, and save the spec to ./cancel-spec.md (the agent cannot write files, so the session saves its output). The spec must cover every path in the current code that runs as part of a cancellation or acts on the same subscription while a cancellation is under way, naming the file each path lives in. Include a state table and a cleanup inventory.
