---
type: llm
focus: { source: file, path: architecture.md }
---

All claims are required.
PASS if architecture.md covers the stated authentication requirements: SAML for staff, an email-link path for contractors, 8-hour sessions revocable within 1 minute, and an append-only audit log with 400-day retention.
PASS if architecture.md says it used docs/auth-requirements.md and set aside docs/billing.md, and contains no billing terms (net-45, credit notes) as design inputs.
PASS if the implementation plan puts the riskiest part (such as revocation within 1 minute, or SAML integration) before low-risk work, and each task is a testable vertical slice.
FAIL if design decisions list no alternatives considered, or the sections for risks and implementation plan are missing.
