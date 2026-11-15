---
type: llm
focus: { source: file, path: migration-plan.md }
---

All claims are required.
PASS if the complexity is judged from the worst factor and names the factors that drive it (for example session pickle data, the ledger logic, or the C extensions), not an averaged score.
PASS if it commits to one strategy, and that strategy uses the side-by-side deployment (for example parallel run or strangler), with a reason.
PASS if the rollback thresholds are defined relative to a baseline that is measured before the migration starts (and it adds a step to measure it), rather than invented fixed numbers.
PASS if it calls for comparison or characterization tests on the ledger module before it is migrated, and handles the pickled sessions as a separate data risk.
FAIL if the timeline is a calendar guess with no tie to measured scope, or if the rollback thresholds are fixed percentages with no source.
