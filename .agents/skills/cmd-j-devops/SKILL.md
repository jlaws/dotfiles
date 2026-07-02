---
name: cmd-j-devops
description: "DevOps consultation — CI/CD, containers, IaC, and observability. Use when configuring Docker, Kubernetes, Terraform, GitHub Actions, or monitoring. Do NOT use for cloud provider selection (use /j-cloud instead)."
disable-model-invocation: true
---

# DevOps Consultation

Before starting, gather diagnostic context:

1. **Detect CI/CD platform** from config files (.github/workflows/, .gitlab-ci.yml, Jenkinsfile, .circleci/).
2. **Identify containerization** by searching for Dockerfile, docker-compose.yml, or container registry configs.
3. **Check IaC setup** for Terraform (.tf files), Pulumi, Helm charts, or Kubernetes manifests.
4. **Get scope overview** of the target area (if the user's provided input specifies a component, scope to that; otherwise scan for infra/, deploy/, .github/, k8s/, or terraform/ directories).

For deep DevOps guidance, delegate to the `devops-engineer` agent, passing the diagnostic findings above and the request. It loads its skills and the `.agents/references/devops/` library (plus `.agents/references/workflow/release-versioning`), then returns specific guidance. Verify its output before presenting.

Help with: the user's provided input
