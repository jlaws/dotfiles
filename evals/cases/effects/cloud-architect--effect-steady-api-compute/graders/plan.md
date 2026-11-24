---
type: llm
focus: { source: file, path: compute-plan.md }
---

All claims are required.
PASS if compute-plan.md puts the image thumbnail API on containers, not Lambda, giving its steady load or its daily request volume as the reason.
PASS if compute-plan.md puts the payment webhook receiver on Lambda.
PASS if compute-plan.md sets the point to move a service from Lambda to containers at about 1 million requests (invocations) per day.
FAIL if compute-plan.md runs the 25-minute nightly reconciliation job as a single Lambda invocation.
