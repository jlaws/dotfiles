---
type: llm
focus: { source: file, path: docs/adr/payments/client-retry-policy.md }
---

All claims are required.
PASS if the ADR file opens with YAML frontmatter holding status accepted, topic payments, created, and updated.
PASS if the ADR states the decision and its rationale, and has a Ruled Out section with a table that lists at least two rejected options (for example retrying every request, or no retries) each with a reason.
FAIL if the ADR uses a status other than accepted, such as superseded or deprecated.
