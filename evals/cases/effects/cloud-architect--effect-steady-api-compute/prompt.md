---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "run steady services or those above about 1M invocations per day on containers, and keep Lambda for bursty work averaging under 100 ms and finishing within 15 minutes; .claude/references/cloud/serverless-patterns.md; hard because a short handler priced per request looks cheap on Lambda, so a quick cost estimate points the steady service at Lambda"
---

Use the kb:cloud-architect agent to choose the compute for three AWS workloads we are moving off a single VM. Each goes either on Lambda or on containers (ECS on Fargate). (1) An image thumbnail API that serves a steady 4 million requests per day around the clock, averaging 80 ms per request with a maximum of 2 seconds. (2) A payment webhook receiver that gets about 20,000 requests per day, arriving in bursts after each billing run, averaging 60 ms. (3) A nightly reconciliation job that takes about 25 minutes from start to finish. Save ./compute-plan.md with the choice for each workload and the reason, and the daily request volume at which you would move a service from Lambda to containers.
