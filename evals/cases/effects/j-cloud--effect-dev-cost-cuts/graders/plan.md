---
type: llm
focus: { source: file, path: cost-plan.md }
---

All claims are required.
PASS if cost-plan.md has exactly three ranked changes, each with an action, a dollar saving estimate, and a risk.
PASS if one change schedules shutdown of the dev environments outside working hours or right-sizes them (for example to smaller instances).
PASS if one change addresses NAT Gateway data charges, for example with VPC gateway endpoints for S3 and DynamoDB.
PASS if one change adds an S3 lifecycle policy that transitions or expires old logs, and the plan names a hidden cost such as retrieval, request, or transition charges.
FAIL if any dollar estimate is missing or the ranking is not ordered by estimated saving.
