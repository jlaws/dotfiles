---
type: llm
focus: { source: file, path: threat-model.md }
---

All claims are required.
PASS if threat-model.md identifies replay of a validly signed event (no timestamp, no processed-event-id check) as a threat, with a mitigation such as a signed timestamp window or idempotency on event id.
PASS if threat-model.md flags that logging full request bodies exposes customer email and card data (information disclosure) and proposes redacting or not logging it.
PASS if every threat carries a risk score of impact x likelihood with both factors in 1 to 4.
FAIL if any of the six STRIDE categories has no section, or the compliance mapping table is missing.
