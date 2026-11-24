---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "coordinate a saga of four or more steps by orchestration, not choreography, and compensate only the completed steps in reverse order; .claude/references/architecture/distributed-communication-patterns.md; hard because the platform already runs a two-step flow by Kafka choreography, and matching the existing pattern looks consistent"
---

/kb:j-arch we are adding order fulfillment to the platform described in ./docs/architecture.md; the steps are in ./docs/fulfillment-flow.md. Decide how the services coordinate this flow and how the steps already done get undone when a later step fails, and save the decision with its reasons to ./saga.md.
