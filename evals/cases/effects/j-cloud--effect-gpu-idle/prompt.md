---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "replace always-on GPUs with per-job capacity, pair spot with checkpointing, and alert on idle GPUs at a low utilization threshold over a time window, per the cloud references; .claude/commands/j-cloud.md; hard because the idle alert needs a specific threshold and window that general cost advice leaves open"
---

/kb:j-cloud reduce GPU spend. A team keeps four p4d.24xlarge instances running 24/7, but their training jobs only use GPUs about 3 hours a day. Jobs run 6 to 10 hours and a lost run costs a day of work. Save the plan to ./gpu-plan.md. It must say how jobs acquire and release GPUs, how interruptions are survived if spot capacity is used, and what alert catches idle GPUs, with its threshold and time window. Write the alert condition on one line in the form `GPU utilization < N% for <window>`, with your threshold and window filled in.
