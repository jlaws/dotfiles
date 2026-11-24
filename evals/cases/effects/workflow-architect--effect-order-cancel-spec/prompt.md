---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "map every branch including compensation in reverse order, give every created resource a Destroyed By and a timeout, give each non-terminal state an outbound transition, and specify the handoff contract; .claude/agents/workflow-architect.md; hard because cancellation can land at any step, so each resource created so far needs its own release"
---

Use the kb:workflow-architect agent to specify, before any code exists, the "customer cancels an order after payment" workflow. The steps are: place a payment hold, reserve inventory, create a shipping label, charge the card, then send a confirmation email. Cancellation can arrive at any of these steps. Save the spec to ./cancel-spec.md. It must contain: a workflow tree with a branch for the happy path, timeout, transient failure with retries, permanent failure, partial failure with compensation order, and concurrent conflict; a cleanup inventory table with the columns Resource, Created By, Destroyed By, Cleanup Method, and Timeout covering every resource the workflow creates; a state table with each state, its valid transitions, and its guard condition; and a handoff contract for the payment gateway call that gives the idempotency key, retry policy, timeout budget, error codes, and dead letter queue.
