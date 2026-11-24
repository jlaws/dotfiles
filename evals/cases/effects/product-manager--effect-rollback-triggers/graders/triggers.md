---
type: llm
focus: { source: file, path: rollback-triggers.md }
---

All claims are required.
PASS if rollback-triggers.md triggers on an error rate above 2x the baseline with an automatic rollback through the feature flag.
PASS if rollback-triggers.md triggers on a p99 latency above 2x the baseline with an investigation first, and a rollback only if there is no fix within 30 minutes.
PASS if rollback-triggers.md triggers on more than 10 user complaints in the first hour by disabling the feature for new users and investigating, not by a full rollback.
FAIL if any data integrity problem does not trigger an immediate rollback, for example because the trigger waits for a count, a rate, or an investigation first.
