---
type: llm
focus: { source: file, path: diagnosis.md }
---

All claims are required.
PASS if the diagnosis separates what the log shows (observed facts such as pool 20/20 in use, a 41230ms connection hold, and the 502 only from ledger-sync-1) from what is inferred.
PASS if it lists at least three distinct hypotheses and gives, for each, evidence that would confirm it and evidence that would refute it.
PASS if it states plainly that the root cause is not yet established, or labels its leading explanation a hypothesis, rather than presenting one cause as proven.
PASS if it names a concrete next step that observes or reproduces the symptom (for example instrumenting connection acquire and release with a unique log prefix, or a load loop that reproduces the pool exhaustion) before any fix is applied.
FAIL if it recommends raising the pool size, the timeout, or adding a retry as the fix without that evidence.
