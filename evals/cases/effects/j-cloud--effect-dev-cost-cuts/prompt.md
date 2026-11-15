---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-cloud cut this AWS bill. Current monthly spend: 8 developer environments on m5.2xlarge instances running 24/7 (about $4,300), a NAT Gateway moving roughly 20 TB a month of mostly S3 and DynamoDB traffic from private subnets, and an S3 bucket of 60 TB of logs that nobody reads after 30 days. Save a plan to ./cost-plan.md with the top three changes ranked by estimated monthly saving, each with a concrete action, an estimated monthly saving in dollars, and one risk. The plan must also name any cost that is easy to miss when making the change.
