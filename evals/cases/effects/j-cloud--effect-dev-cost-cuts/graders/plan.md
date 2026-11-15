---
type: llm
focus: { source: file, path: cost-plan.md }
---

All claims are required.
PASS if cost-plan.md has exactly three ranked changes, each with an action, a dollar saving estimate, and a risk.
PASS if each change is a defensible saving for the bill described (for example scheduling or right-sizing the dev instances, VPC gateway endpoints for the NAT traffic, or an S3 lifecycle policy for the old logs) with a dollar estimate plausible for the stated spend.
PASS if the plan names at least one easy-to-miss cost tied to one of its changes (for example lifecycle transition or request charges, minimum storage duration, retrieval fees, or traffic that still crosses the NAT Gateway).
FAIL if any dollar estimate is missing or the ranking is not ordered by estimated saving.
