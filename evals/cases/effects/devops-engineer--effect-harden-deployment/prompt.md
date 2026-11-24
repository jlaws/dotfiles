---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "harden a Deployment with non-root, no privilege escalation, a memory limit without a CPU limit, a process-only liveness probe, a dedicated ServiceAccount with no token, default-deny ingress, and a pinned image; .claude/agents/devops-engineer.md; hard because the seeded liveness probe checks dependencies and common guidance puts a CPU limit on every container"
---

Use the kb:devops-engineer agent to harden the Kubernetes Deployment in ./deploy.yaml and save the corrected manifests to ./deploy.fixed.yaml (leave deploy.yaml unchanged). The API is latency-sensitive. On this service /healthz only checks the process, while /healthz/full also pings postgres and redis. Requirements: pods never run as root and cannot gain privileges; memory is bounded so a leak cannot take down the node, and CPU throttling must not add latency; a brief postgres or redis outage must not restart pods, though it may take them out of the Service; the pod uses a dedicated ServiceAccount (include the ServiceAccount object) whose token is not mounted; the namespace denies all ingress by default, with a NetworkPolicy that allows only pods labelled app=gateway to reach port 8080; the image is pinned to tag 1.8.2, not latest. Put every object in the one file as separate YAML documents.
