---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "for a bootstrapped product under 1M dollars ARR choose PostHog, start the tracking plan with 10 to 15 core-funnel events, and remove an event once it goes unqueried for 90 days; .claude/references/business/analytics-instrumentation.md; hard because the provider choice by funding stage and the 90-day removal window are house rules, and migrating every existing event looks safest"
---

Use the kb:business-analyst agent to plan the analytics reset for our B2B SaaS product. We are bootstrapped with no outside funding, our ARR is about $300,000, and our customers are small businesses in the US. Today we log about 400 event types, including every click, hover, and scroll, into a self-built events table that nobody can query quickly, and we want to move to an analytics product. Save the plan to ./analytics-plan.md. It must cover: which analytics provider to use and why; how many events the first version of the new tracking plan has, with the list of them; and the rule for when an event is removed from the plan later.
