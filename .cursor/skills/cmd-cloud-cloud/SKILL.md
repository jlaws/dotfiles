---
name: cmd-cloud-cloud
description: "Cloud architecture consultation — cloud infrastructure design, cost optimization, and deployment planning. Use when designing cloud infrastructure, optimizing costs, or planning deployments. Do NOT use for container/Kubernetes questions (use /cmd-devops-devops instead)."
disable-model-invocation: true
---

# Cloud Architecture Consultation

Before starting, gather diagnostic context:

1. **Detect cloud provider** from config files (AWS: .aws/, CDK, SAM, CloudFormation; GCP: app.yaml, .gcloudignore; Azure: azure-pipelines.yml, bicep).
2. **Identify IaC patterns** by searching for Terraform (.tf), Pulumi, CDK, CloudFormation templates, or Helm charts.
3. **Check IaC tooling config** (terraform.tfvars, terragrunt.hcl, pulumi.yaml, serverless.yml).
4. **Get scope overview** of the target area (if the user specifies a service, scope to that; otherwise scan for infra/, deploy/, cdk/, terraform/ directories).

Help with the cloud architecture topic specified by the user.
