---
type: llm
focus: { source: file, path: cancel-spec.md }
---

All claims are required.
PASS if cancel-spec.md includes, as steps of the cancellation, the database trigger in migrations/0007_status_audit.sql that writes a billing_events row when the status changes and the poller in app/billing_sync.py that reads that row and cancels the plan at the payment provider.
PASS if cancel-spec.md includes the nightly renewal job in app/scheduler.py and specifies what happens when a cancellation lands while that job is renewing or charging the same subscription.
PASS if the cleanup inventory includes the subscription at the payment provider and names the path that ends it there (the billing_sync poller calling the provider).
FAIL if cancel-spec.md treats the cancellation as finished once the status is set to cancelled and the email is queued, with no step that stops billing at the payment provider.
