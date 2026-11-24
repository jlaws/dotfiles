---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "for production services under 1k requests per second, sample 10 to 50 percent probabilistically with a ParentBased sampler and force-keep error and high-latency traces; .claude/references/devops/observability.md; hard because the cost pressure pulls toward the 1 to 5 percent rates common for high-traffic systems"
---

/kb:j-observe tracing costs for our checkout services are too high; the facts are in ./docs/services.md. Choose the trace sampling for each environment, covering the production sampling rate, how a sampling decision carries across service hops, and what happens to the traces of failed and slow requests. Save it to ./sampling.md.
