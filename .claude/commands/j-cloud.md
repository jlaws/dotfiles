---
name: j-cloud
description: "Cloud architecture consultation. Use when designing cloud infrastructure, optimizing costs, or planning deployments. Do NOT use for container/Kubernetes questions (use /j-devops instead)."
argument-hint: "<question-or-task>"
---

Load skill `analysis-output-patterns` for output structure rules.
Load skill `design-first` for design-before-implementation discipline.

Before starting, gather diagnostic context:

1. **Detect cloud provider** from config files (AWS: .aws/, CDK, SAM, CloudFormation; GCP: app.yaml, .gcloudignore; Azure: azure-pipelines.yml, bicep).
2. **Identify IaC patterns** by searching for Terraform (.tf), Pulumi, CDK, CloudFormation templates, or Helm charts.
3. **Check IaC tooling config** (terraform.tfvars, terragrunt.hcl, pulumi.yaml, serverless.yml).
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a service, scope to that; otherwise scan for infra/, deploy/, cdk/, terraform/ directories).

Load relevant references based on the diagnostic context:
- `references/cloud/cost-optimization` -- rightsizing, spot/reserved, egress, budget guardrails
- `references/cloud/serverless-patterns` -- functions, cold starts, event-driven, step orchestration
- `references/cloud/multi-cloud-architecture` -- portability, provider abstraction, multi-region topology
- `references/cloud/file-storage-patterns` -- object storage, CDN, signed URLs, lifecycle policies
- `references/cloud/gpu-compute-management` -- GPU provisioning, scheduling, cost control for ML workloads

Help with: $ARGUMENTS
