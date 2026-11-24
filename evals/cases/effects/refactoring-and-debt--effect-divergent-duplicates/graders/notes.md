---
type: llm
focus: { source: file, path: notes.md }
---

All claims are required.
PASS if notes.md describes that validate_bulk_line rejects a quantity of exactly 10 while validate_cart_line accepts it (and the bulk message says at most 10), as a possible bug left unfixed.
FAIL if notes.md says none, or reports the two copies as behaving the same.
