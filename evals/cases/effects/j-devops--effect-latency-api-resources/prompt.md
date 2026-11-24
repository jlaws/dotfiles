---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "for a latency-sensitive service set CPU and memory requests, omit the CPU limit, and set the memory limit at 1.5 to 2 times the request; .claude/references/devops/kubernetes-configuration.md; hard because common Kubernetes guidance puts a CPU limit on every container, and the seeded manifest sets limits equal to requests"
---

/kb:j-devops set the CPU and memory resources for the api container in ./deploy/checkout-api.yaml, using the service facts in ./docs/checkout-api.md. Edit the manifest in place and explain each value in ./resources-notes.md.
