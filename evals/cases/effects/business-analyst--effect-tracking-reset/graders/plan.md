---
type: llm
focus: { source: file, path: analytics-plan.md }
---

All claims are required.
PASS if analytics-plan.md chooses PostHog as the analytics provider.
PASS if the first version of the new tracking plan in analytics-plan.md has between 10 and 15 events, tied to the product's core funnel.
PASS if analytics-plan.md removes an event from the plan once it has gone unqueried for 90 days.
FAIL if analytics-plan.md carries the current click, hover, or scroll events into the new plan as a whole, or keeps the 400 existing event types.
