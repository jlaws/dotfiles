---
type: llm
focus: { source: file, path: import-spec.md }
---

All claims are required.
PASS if import-spec.md has all four views (by workflow step, by component, by user journey, by state) and they are consistent with each other.
PASS if each system state in the state table has a customer-facing label and an operator view, and any state without a customer label is merged or justified.
PASS if the concurrent-upload path detects the conflict (optimistic lock, version check, or per-account lock) and returns 409 or retries with fresh state instead of overwriting.
PASS if the cleanup table names who destroys the temp file and the partial rows after a worker crash (for example a TTL, a resume, or a compensating delete) with a timeout.
FAIL if the concurrent-upload path lets the second upload silently overwrite the first.
