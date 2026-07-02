---
name: j-cloud
description: "Cloud architecture consultation. Use when designing cloud infrastructure, optimizing costs, or planning deployments. Do NOT use for container/Kubernetes questions (use /j-devops instead)."
argument-hint: "<question-or-task>"
---

Before starting, gather diagnostic context:

1. **Detect cloud provider** from config files (AWS: .aws/, CDK, SAM, CloudFormation; GCP: app.yaml, .gcloudignore; Azure: azure-pipelines.yml, bicep).
2. **Identify IaC patterns** by searching for Terraform (.tf), Pulumi, CDK, CloudFormation templates, or Helm charts.
3. **Check IaC tooling config** (terraform.tfvars, terragrunt.hcl, pulumi.yaml, serverless.yml).
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a service, scope to that; otherwise scan for infra/, deploy/, cdk/, terraform/ directories).

For deep cloud guidance, delegate to the `cloud-architect` agent, passing the diagnostic findings above and the request. It loads its skills (design-first) and the `.agents/references/cloud/` library, then returns specific guidance. Verify its output before presenting.

Help with: $ARGUMENTS
