---
type: llm
focus: { source: file, path: notification-plan.md }
---

All claims are required.
PASS if notification-plan.md sends the service outage alert on push, SMS, and email all at the same time.
PASS if notification-plan.md sends the payment failed notice on push and email, with email following when push is still undelivered after 5 minutes.
PASS if notification-plan.md sends the comment reply on push or in-app, with an email digest as the fallback when it is still unread after 24 hours.
FAIL if notification-plan.md gives the weekly product tips message any channel other than email, or any fallback.
