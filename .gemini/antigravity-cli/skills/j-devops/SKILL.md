---
name: j-devops
description: "DevOps consultation: CI/CD, containers, IaC, and observability. Use when configuring Docker, Kubernetes, Terraform, GitHub Actions, or monitoring. Do NOT use for cloud provider selection (use /j-cloud instead)."
---

# Devops

Load skill `analysis-output-patterns` for output structure rules.

Before starting, gather diagnostic context:

1. **Detect CI/CD platform** from config files (.github/workflows/, .gitlab-ci.yml, Jenkinsfile, .circleci/).
2. **Identify containerization** by searching for Dockerfile, docker-compose.yml, or container registry configs.
3. **Check IaC setup** for Terraform (.tf files), Pulumi, Helm charts, or Kubernetes manifests.
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a component, scope to that; otherwise scan for infra/, deploy/, .github/, k8s/, or terraform/ directories).

Load the relevant references for the topic, based on the diagnostic context:

Reference library at `~/.agents/references/devops/`:
- docker-patterns, github-actions-patterns, gitops-workflow
- incident-management, incident-readiness, kubernetes-configuration, monorepo-tools
- observability, pipeline-design, security-policies
- sre-practices, terraform-module-library

Also see `~/.agents/references/workflow/`:
- release-versioning

For deep DevOps guidance, you may instead delegate to the `devops-engineer` agent, passing the diagnostic findings above and the request. It loads its skills and the `~/.agents/references/devops/` library (plus `~/.agents/references/workflow/release-versioning`), then returns specific guidance. Verify its output before presenting.

Help with: $ARGUMENTS
