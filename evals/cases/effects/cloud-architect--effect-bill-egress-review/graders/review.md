---
type: llm
focus: { source: file, path: cost-review.md }
---

All claims are required.
PASS if the S3 LIST request cost is addressed by cutting the list calls (for example an inventory or index, prefix design, or caching the listing).
PASS if the NAT Gateway cost is addressed with an S3 gateway VPC endpoint (or keeping the traffic off the NAT), and the internet egress cost is addressed with a CDN such as CloudFront in front of the bucket.
PASS if the steady RDS instance and steady EC2 fleet are moved to reserved capacity or a savings plan rather than spot.
FAIL if the review recommends removing or shrinking the eu-west-1 replication.
