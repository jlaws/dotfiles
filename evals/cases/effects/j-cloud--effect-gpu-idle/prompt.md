---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-cloud reduce GPU spend. A team keeps four p4d.24xlarge instances running 24/7, but their training jobs only use GPUs about 3 hours a day. Jobs run 6 to 10 hours and a lost run costs a day of work. Save the plan to ./gpu-plan.md. It must say how jobs acquire and release GPUs, how interruptions are survived if spot capacity is used, and what alert catches idle GPUs, with its threshold and time window.
