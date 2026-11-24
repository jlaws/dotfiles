---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "retry only 408, 429, and 5xx, never 400, 401, or 404, honor Retry-After, send one idempotency key on every POST retry, and record the choice as an accepted ADR at docs/adr/<topic>/<slug>.md with a Ruled Out table; .claude/agents/architecture-specialist.md; hard because the agent's ADR rule and the retry reference fix details, such as the status that never changes from accepted and Retry-After overriding backoff, that a model would otherwise leave to taste"
---

Use the kb:architecture-specialist agent to design the client retry policy for our payments SDK. The SDK calls POST /charges, GET /charges/{id}, and DELETE /charges/{id}. Responses seen in production include 400, 401, 404, 408, 409, 429 (with a Retry-After header), 500, 502, 503, and 504. Save the design to ./retry-design.md. It must say which status codes are retried and which are not, how POST /charges is made safe to retry, how Retry-After and backoff jitter are handled, and the attempt limit. Also record the decision as an ADR at docs/adr/payments/client-retry-policy.md: the path is topic then slug with no number in the file name, YAML frontmatter holds status (accepted), topic (payments), created, and updated, and a "Ruled Out" section holds a table of the rejected options with the reason for each.
