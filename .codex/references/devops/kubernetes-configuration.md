# Kubernetes Configuration

## Critical Rules

- **Never** run containers as root -- set `runAsNonRoot: true` and `allowPrivilegeEscalation: false` on every pod
- **Always** set memory requests/limits -- pods without limits can OOMKill neighbors and crash nodes
- **Never** use the `default` ServiceAccount -- create dedicated SAs with `automountServiceAccountToken: false`
- **Always** apply default-deny NetworkPolicies per namespace before adding allow rules
- **Never** make liveness probes check external dependencies -- a database blip will cascade-restart all pods

## Manifest Generation Workflow

1. Gather requirements (stateless/stateful, ports, storage, scaling, health endpoints)
2. Create Deployment/StatefulSet
3. Create Service (ClusterIP/LoadBalancer/NodePort)
4. Add ConfigMap and/or Secret
5. Add PVC if stateful
6. Apply security context + pod security standards
7. Add NetworkPolicy (default-deny + allow-list)
8. Add standard labels
9. Validate with `kubectl apply --dry-run=server`

## Resource Limits

| Workload Type | Requests | Limits |
|---------------|----------|--------|
| API server | 250m / 256Mi | 500m / 512Mi |
| Worker/queue consumer | 500m / 512Mi | 1000m / 1Gi |
| Batch job | 1000m / 1Gi | 2000m / 2Gi |
| Sidecar (envoy, etc.) | 100m / 128Mi | 200m / 256Mi |

**Rules:**
- Always set requests. Limits optional but recommended for memory.
- CPU limits cause throttling -- omit for latency-sensitive services; use only requests.
- Memory limits 1.5-2x requests. OOMKill is worse than throttling.
- Never set requests = limits unless you want Guaranteed QoS (rare).

## Health Probes

- **startupProbe**: Slow-starting apps (JVM, ML models). `failureThreshold * periodSeconds` = max startup time.
- **livenessProbe**: Detect deadlocks. Conservative: `failureThreshold: 3`, `periodSeconds: 10`. Don't check dependencies.
- **readinessProbe**: Gate traffic. Check app health + critical dependencies. `periodSeconds: 5`.

**Common mistake**: Liveness probe checks database connectivity -> DB blip restarts all pods -> cascading failure.

## Security Context (Always Apply)

```yaml
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop: [ALL]
    volumeMounts:
    - name: tmp
      mountPath: /tmp
  volumes:
  - name: tmp
    emptyDir: {}
```

- `readOnlyRootFilesystem: true` requires `/tmp` emptyDir mount for temp files
- Drop ALL capabilities, add back only what's needed (almost never)
- Never run as root. If image requires it, fix the image.
- `seccompProfile: RuntimeDefault` blocks dangerous syscalls with zero app changes.

## Pod Security Standards

Apply via namespace labels. Three tiers:

| Level | Use Case | Key Restrictions |
|-------|----------|-----------------|
| `privileged` | System workloads (CNI, monitoring agents) | None |
| `baseline` | Most workloads | No hostNetwork, no privileged containers, no hostPath |
| `restricted` | Sensitive workloads | Must: runAsNonRoot, drop ALL caps, seccompProfile, readOnlyRootFilesystem |

**Default**: `restricted` for app namespaces, `baseline` for infrastructure, `privileged` only for system.

```yaml
metadata:
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

Use `audit` + `warn` at stricter level than `enforce` during migration.

## Network Policies

**Strategy: Default-Deny + Allow-List**

1. Apply default-deny to every namespace
2. Allow DNS egress (everything breaks without this)
3. Allow specific service-to-service communication via label selectors

**Rules:**
- Always deny by default. Allowlists > denylists.
- Use `namespaceSelector` + `podSelector` together for cross-namespace policies.
- Requires a CNI that supports it (Calico, Cilium). Default kubenet does NOT enforce.
- Test in staging with `audit` mode before enforcing.

See `references/security-policies.md` for NetworkPolicy YAML examples.

## RBAC

- **Least privilege**: Start with zero permissions, add only what's needed
- **Namespace-scoped Roles over ClusterRoles**: Minimize blast radius
- **Dedicated ServiceAccounts**: Never use `default` SA. Set `automountServiceAccountToken: false` unless needed.
- **No wildcard verbs**: `"*"` on resources = admin. Always enumerate specific verbs.

| Role | Resources | Verbs |
|------|-----------|-------|
| Pod reader | pods | get, list, watch |
| Deployment manager | deployments, pods | get, list, watch, create, update, patch, delete |
| CI/CD deployer | deployments, services, configmaps | get, list, create, update, patch |
| Secret reader (scoped) | secrets (by resourceNames) | get |

**Key opinion**: CI/CD service accounts should NOT have `delete` on pods -- let the deployment controller handle pod lifecycle.

## Labels (Standard K8s Labels)

```yaml
app.kubernetes.io/name: <app>
app.kubernetes.io/version: "<version>"
app.kubernetes.io/component: backend|frontend|worker
app.kubernetes.io/part-of: <system>
app.kubernetes.io/managed-by: helm|kustomize|kubectl
```

## Service Selection

| Type | When |
|------|------|
| `ClusterIP` | Internal services (default) |
| `LoadBalancer` | External-facing, cloud provider LB |
| `NodePort` | Development only, never production |
| `ClusterIP: None` (headless) | StatefulSets needing stable DNS |

## Secret Management

- Never commit plaintext secrets to Git
- Use: External Secrets Operator (AWS SM/Vault), Sealed Secrets, or SOPS
- `stringData` in manifests is for local dev only

## Manifest Organization

**Kustomize** (default choice):
```
base/            # Shared manifests
overlays/
  dev/           # Dev overrides (replica count, resource limits)
  prod/          # Prod overrides
```

**Helm** (when you need templating + packaging):
- Use `values.schema.json` to validate inputs
- Template helpers in `_helpers.tpl` -- keep DRY
- Always support `resources`, `nodeSelector`, `tolerations`, `affinity` in values
- Use `helm template --dry-run --debug` to verify before install

### Helm Anti-Patterns
- Overly complex templates with deeply nested conditionals
- Not pinning chart dependencies to exact versions
- Using `helm install` without `--atomic` (leaves partial deploys)
- Storing secrets in values files

## Environment-Specific Rules

| Rule | Dev | Staging | Prod |
|------|-----|---------|------|
| Debug mode allowed | Yes | No | No |
| HTTPS required | No | Yes | Yes |
| Min replicas | 1 | 2 | 3 |
| Resource limits required | No | Yes | Yes |
| Secret encryption required | No | No | Yes |

**Enforce in CI**: validate configs against environment-specific rulesets before merge.

## OPA Gatekeeper

Use for policies Pod Security Standards can't enforce:
- Required labels on all resources
- Image registry restrictions (only pull from approved registries)
- Resource limit requirements
- Naming conventions

**Opinion**: Start with `dryrun` enforcement action, promote to `deny` after review.

## Service Mesh mTLS

- `STRICT` in production -- reject all non-mTLS traffic
- `PERMISSIVE` during mesh rollout -- accept both
- Use `AuthorizationPolicy` for fine-grained service-to-service access control

## Validation Pipeline

```bash
kubectl apply -f manifest.yaml --dry-run=server  # API server validation
kube-linter lint manifest.yaml                     # Best practices
kube-score score manifest.yaml                     # Security scoring
```

Run all three in CI before merge.

## Security Checklist

- [ ] Default-deny NetworkPolicy in all namespaces
- [ ] DNS egress allowed
- [ ] Pod Security Standards enforced (restricted for apps)
- [ ] Dedicated ServiceAccounts per workload
- [ ] `automountServiceAccountToken: false` where not needed
- [ ] No containers running as root
- [ ] All capabilities dropped
- [ ] Read-only root filesystem
- [ ] Seccomp profile enabled
- [ ] RBAC follows least privilege
- [ ] Image pull from approved registries only

## References

- .codex/references/cloud/gpu-compute-management.md — GPU scheduling and resource management
- .codex/references/cloud/cost-optimization.md — cluster cost optimization strategies
