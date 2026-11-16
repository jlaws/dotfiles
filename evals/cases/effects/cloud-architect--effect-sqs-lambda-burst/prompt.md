---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:cloud-architect agent to design an AWS serverless pipeline that processes uploaded video files: an S3 upload event goes to an SQS queue, and Lambda consumes the queue in batches of 10. Requirements: SQS delivers at least once and Lambda retries after a timeout, so processing a message twice must not create duplicate output; when one message in a batch fails, the other nine must not be reprocessed; this account runs other Lambda functions that share the default regional concurrency limit, and a burst of 50,000 uploads must not throttle them; some files take about 40 minutes to process. Save the design to ./design.md, including an infrastructure-as-code snippet (Terraform, CloudFormation, or SAM) for the queue event source mapping and the function's concurrency settings, and the handler's batch-failure return shape.
