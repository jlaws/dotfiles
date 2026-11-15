---
type: llm
focus: { source: file, path: resolution.md }
---

All claims are required.
PASS if the top of resolution.md classifies the issue with one of P0, P1, P2 or P3 and gives a reason tied to user impact or a workaround.
FAIL if the classification is missing, or is P0 for a silently ignored id with no data loss or outage.
