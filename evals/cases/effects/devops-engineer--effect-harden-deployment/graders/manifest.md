---
type: llm
focus: { source: file, path: deploy.fixed.yaml }
---

All claims are required.
PASS if the livenessProbe targets /healthz (the process-only endpoint) and not /healthz/full, while the readinessProbe may keep /healthz/full.
PASS if the container sets a memory limit and has no CPU limit, and sets allowPrivilegeEscalation to false.
PASS if a NetworkPolicy denies all ingress by default and another rule or policy allows only pods labelled app=gateway to port 8080, and the Deployment references a dedicated serviceAccountName.
FAIL if the image still uses the latest tag or is not shop/api:1.8.2.
