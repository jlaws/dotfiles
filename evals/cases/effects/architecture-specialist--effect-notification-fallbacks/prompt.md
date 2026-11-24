---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "route each notification by urgency, with outage alerts on push, SMS, and email at once, payment notices on push and email with email after 5 minutes undelivered, activity on push or in-app with an email digest after 24 hours unread, and marketing tips on email with no fallback; .claude/references/architecture/notification-systems.md; hard because the fallback timings are house thresholds a model would otherwise pick freely, and a sequential escalation for outages looks prudent"
---

Use the kb:architecture-specialist agent to choose the delivery channels for four notification types in our consumer app: a service outage alert, a payment failed notice, a reply to the user's comment, and a weekly product tips message. The app can send push, SMS, email, and in-app messages. For each type, say which channels it is sent on, whether those channels go out at the same time or one after another, and the fallback rule, including how long to wait before the fallback fires (or that there is no fallback). Save the plan to ./notification-plan.md.
