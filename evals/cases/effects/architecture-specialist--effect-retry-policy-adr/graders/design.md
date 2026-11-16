---
type: llm
focus: { source: file, path: retry-design.md }
---

All claims are required.
PASS if retry-design.md retries 408, 429, 500, 502, 503, and 504 and explicitly does not retry 400, 401, 404, or 409.
PASS if retry-design.md makes POST /charges retryable only with an idempotency key sent on every attempt of the same logical charge.
PASS if retry-design.md says a 429 waits at least the Retry-After value and that backoff is exponential with jitter, with a stated maximum attempt count.
FAIL if retry-design.md retries POST /charges without any idempotency key.
