---
type: llm
focus: { source: file, path: design.md }
---

All claims are required.
PASS if design.md caps the consumer's concurrency at a finite number (reserved concurrency or the event source mapping's maximum concurrency) and ties it to protecting the other functions that share the account limit.
PASS if design.md says a single Lambda invocation cannot run 40 minutes (15-minute maximum) and moves the long files to another mechanism such as Step Functions, ECS/Fargate, or AWS Batch.
PASS if the duplicate protection is concrete: an idempotency key (for example bucket/key/version id or message id) checked with a conditional write, or a deterministic output key.
FAIL if the Lambda timeout is set above 900 seconds anywhere in design.md.
