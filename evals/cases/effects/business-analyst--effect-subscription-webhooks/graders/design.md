---
type: llm
focus: { source: file, path: billing-design.md }
---

All claims are required.
PASS if billing-design.md rejects storing raw card numbers and uses provider tokenization or hosted checkout with a customer and payment method token.
PASS if billing-design.md verifies the webhook signature before any processing and deduplicates by the provider event id, marking the event processed only after the handler succeeds so a thrown error lets the provider retry.
PASS if billing-design.md stores amounts as integers in the smallest currency unit (cents) with a currency code.
PASS if billing-design.md gives the state transitions (including past due to canceled and paused to active) and a dunning schedule with specific retry days and a final cancel step.
FAIL if billing-design.md marks a webhook event processed before the handler succeeds.
