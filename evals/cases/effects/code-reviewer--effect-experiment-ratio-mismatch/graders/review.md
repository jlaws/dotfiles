---
type: llm
focus: { source: file, path: review.md }
---

All claims are required.
PASS if review.md identifies the 10,240 versus 9,760 user split as a sample ratio mismatch against the designed 50/50 allocation (for example by a chi-squared test on the counts).
PASS if review.md says the experiment result cannot be trusted while the mismatch is unexplained, and holds the rollout to 100% until its cause is found.
FAIL if review.md accepts the reported p = 0.02 conversion lift as evidence that supports shipping the change.
