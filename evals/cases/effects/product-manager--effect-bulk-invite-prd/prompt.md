---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "write a PRD that puts unevidenced requests in non-goals, gives leading and lagging metrics with baseline, target, and timeframe, covers edge cases in Given/When/Then, and stages the launch with numeric rollback triggers; .claude/agents/product-manager.md; hard because stakeholders ask for SSO and an export in the same release, which invites building them in"
---

Use the kb:product-manager agent to write a PRD for bulk user invite. Evidence: enterprise admins with more than 50 members must add users one at a time; support sees 37 tickets a month on it and 3 churned accounts cited it. Two stakeholders also asked us to add SSO and an analytics export to the same release, with no data behind either. Save the PRD to ./prd.md. It must contain: the problem with who, what, and evidence; a one-paragraph press release written as if the feature shipped; success metrics with a leading and a lagging indicator, each with a baseline, a target, and a timeframe; non-goals; Given/When/Then acceptance criteria covering a happy path, duplicate emails in the upload, an upload over 500 rows, and the feature flag being off; and a staged launch plan with rollback triggers that carry numeric thresholds.
