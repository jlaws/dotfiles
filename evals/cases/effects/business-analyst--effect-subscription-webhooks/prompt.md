---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "design subscription billing with provider tokenization, signature-verified webhooks deduplicated by event id and marked processed only after the handler succeeds, integer minor-unit amounts, a full state machine, and a dated dunning schedule; .claude/agents/business-analyst.md; hard because marking an event processed before the handler runs looks like correct deduplication and silently drops the provider retry"
---

Use the kb:business-analyst agent to design the billing flow for a subscription product using a hosted payment provider. A teammate proposes storing customers' card numbers in our own database "so renewals are simple". Save the design to ./billing-design.md. It must cover: whether to store card numbers and what to do instead; how a webhook is authenticated before it is processed; how a webhook the provider delivers twice is kept from crediting a customer twice, and what happens to the event record if our handler throws; how money amounts are stored; a subscription state machine with trial, active, past due, paused, and canceled states and their transitions; and a dunning schedule for failed renewals with the day of each retry and when the subscription is canceled.
