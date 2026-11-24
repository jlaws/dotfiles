---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "isolate tenants by tier: a shared database with row-level separation for many small tenants, dedicated infrastructure for HIPAA tenants, and cross-tenant reporting through a separate warehouse path; .claude/references/architecture/saas-multi-tenancy.md; hard because a separate database per regulated tenant looks sufficient, and a report across tenant stores invites a direct cross-database query"
---

/kb:j-system-design design the data tenancy for our clinic scheduling SaaS from ./docs/requirements.md. Save the architecture document to ./architecture.md. For each group of tenants, choose how its data is isolated, and choose where the cross-tenant finance report reads its data from, with the reason for each choice.
