---
name: cloud
description: "Cloud architecture consultation — launches cloud-architect subagent. Use when designing cloud infrastructure, optimizing costs, or planning deployments."
argument-hint: "<question-or-task>"
---

Before invoking the subagent, gather diagnostic context:

1. **Detect cloud provider** from config files (AWS: .aws/, CDK, SAM, CloudFormation; GCP: app.yaml, .gcloudignore; Azure: azure-pipelines.yml, bicep).
2. **Identify IaC patterns** by searching for Terraform (.tf), Pulumi, CDK, CloudFormation templates, or Helm charts.
3. **Check IaC tooling config** (terraform.tfvars, terragrunt.hcl, pulumi.yaml, serverless.yml).
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a service, scope to that; otherwise scan for infra/, deploy/, cdk/, terraform/ directories).

Use the cloud-architect subagent to help with: $ARGUMENTS
