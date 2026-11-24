---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "cut an AWS bill with ranked, priced fixes: fewer S3 LIST calls, a gateway endpoint for NAT traffic, a CDN for internet egress, and reserved capacity for the steady fleet and RDS, while keeping the compliance replica; .claude/agents/cloud-architect.md; hard because spot looks like the biggest compute saving and the replication line looks like an easy cut"
---

Use the kb:cloud-architect agent to review this monthly AWS bill (us-east-1 primary) and cut cost. Line items: 6 x m5.xlarge on-demand running 24/7 behind a steady API, $840; NAT Gateway data processing 22,000 GB, $990; data transfer out to the internet 9,000 GB, $810 (public image assets are served straight from an S3 bucket); S3 LIST requests 400,000,000, $2,000; S3 cross-region replication transfer to eu-west-1 8,000 GB, $160 (compliance requires the eu-west-1 replica to stay); CloudWatch Logs ingestion 1,800 GB, $900 (debug logging left on); 40 unattached 500 GB gp3 EBS volumes, $1,600; 12 unassociated Elastic IPs, $43; RDS db.r5.2xlarge Multi-AZ on-demand 24/7, $1,900. Save the review to ./cost-review.md as a markdown table ranked by estimated monthly saving, highest first, with columns for the line item, the fix, and the estimated saving in dollars, followed by a short list of items you would not cut and why.
