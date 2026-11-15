---
type: llm
focus: { source: file, path: findings.md }
---

All claims are required.
PASS if the vendor 40% claim is marked UNVERIFIED and not used as a finding about our own system.
PASS if it states the missing request counts or sample sizes as a gap, and does not estimate them.
PASS if the improvement is stated with exact figures and units (for example 241 ms in March to 187 ms in April), without vague words such as significantly.
PASS if the conclusion that the migration caused the drop is labeled as an inference, noting that March was a high month and only two post-migration points exist.
FAIL if the vendor claim is presented as a finding or any figure is invented.
