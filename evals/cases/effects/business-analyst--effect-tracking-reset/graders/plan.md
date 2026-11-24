---
type: llm
focus: { source: file, path: analytics-plan.md }
---

All claims are required.
PASS if analytics-plan.md chooses PostHog as the analytics provider.
PASS if the first version of the new tracking plan in analytics-plan.md has between 10 and 15 events, tied to the product's core funnel.
PASS if analytics-plan.md removes an event from the plan once it has gone unqueried for 90 days.
FAIL if analytics-plan.md carries the current click, hover, and scroll events into the new plan wholesale (a single core-funnel event that happens to be a click does not count), or the first version of the plan keeps more than 15 events.
