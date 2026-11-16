---
type: llm
focus: { source: file, path: retry-design.md }
---

All claims are required.
PASS if retry-design.md makes POST /charges safe to retry by sending the same idempotency key on every attempt of one logical charge.
PASS if retry-design.md says a 429 waits at least the Retry-After value before the next attempt.
FAIL if retry-design.md leaves any of 408, 429, 500, 502, 503, or 504 out of the retried codes.
FAIL if retry-design.md marks 400, 401, or 404 as retried, or does not state that each of them is not retried.
