---
name: architecture-workflow-spec
description: "Workflow specification — map all execution paths, failure modes, and handoff contracts before implementation. Use when designing workflows, specifying failure scenarios, or mapping state machines. Do NOT use for system architecture (use /arch instead)."
argument-hint: "<workflow-or-feature-to-specify>"
---

Before starting, gather diagnostic context:

1. **Detect API routes** by searching for route definitions, controllers, or endpoint handlers.
2. **Find background jobs** by searching for worker, job, queue, cron, or scheduler patterns.
3. **Identify event listeners** by searching for event emitters, subscribers, pub/sub patterns, or webhook handlers.
4. **Check for existing state machines** by searching for state, status, transition, or workflow in models/schemas.
5. **Get scope overview** of the target area (if $ARGUMENTS specifies a workflow, scope to that; otherwise scan for the main application entry points).

Address: $ARGUMENTS
