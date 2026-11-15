---
type: llm
focus: { source: file, path: design.md }
---

All claims are required.
PASS if design.md has sections covering cache placement, invalidation, and failure behavior.
PASS if the invalidation approach keeps staleness within the stated 2-second tolerance (for example a TTL of 2 seconds or less, or write-through or event-driven invalidation).
PASS if the failure section says what happens when the cache is unavailable (for example falling back to the database with load protection).
FAIL if any of the three sections is missing, or the design ignores the stated read and write rates.
