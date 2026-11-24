---
type: llm
focus: { source: file, path: runbook.md }
---

All claims are required.
PASS if runbook.md has its own procedure for each of the eight in-scope alerts: HighErrorRate, HighLatencyP95, DiskUsageWarning, DiskUsageCritical, MemoryPressure, ReplicaLagHigh, QueueBacklogWarning, and QueueBacklogCritical.
PASS if each of those eight procedures, including DiskUsageCritical and QueueBacklogCritical, has its own triage steps, commands, and escalation point, with the escalation teams the alert file names for DiskUsageCritical, ReplicaLagHigh, and QueueBacklogCritical.
PASS if CertExpiringSoon appears as a one-line TODO naming issue #88; this TODO is the marker the prompt asked for and is not a placeholder.
FAIL if any of the eight procedures defers to another (for example same as, see above, as for, or similar to) or is a heading with no steps under it.
