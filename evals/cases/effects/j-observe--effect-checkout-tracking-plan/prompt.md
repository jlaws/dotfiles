---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "design an analytics tracking plan with one naming convention, typed properties, enforced PII exclusion, server-side revenue events, no free-text or unbounded properties, and a review process; .claude/commands/j-observe.md; hard because client-side purchase tracking and raw IDs as properties are the default setup"
---

/kb:j-observe design the analytics tracking plan for our web checkout flow (cart viewed, shipping step, payment step, order placed, order failed). The product team will use Mixpanel-style funnels. Save it to ./tracking-plan.md. Requirements: one naming convention for all events and properties, stated at the top; each event lists its properties with types; no personal data (email, name, address, IP) may appear as an event property, with a stated way to enforce that; order placed and revenue must be recorded where ad blockers cannot drop them, and the plan says where each event is emitted (client or server) and why; free-text and unbounded values must not be properties as-is; and the plan says how new events get reviewed so names do not drift.
