---
type: llm
focus: { source: file, path: retry-design.md }
---

All claims are required.
PASS if retry-design.md applies jitter to the backoff delay and says how it is applied, such as a random range or a named jitter method (for example full, equal, or decorrelated jitter).
FAIL if retry-design.md uses no jitter, or mentions jitter without saying how it is applied.
