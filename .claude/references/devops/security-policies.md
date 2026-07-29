# Kubernetes Security Policies

Reference for network policies, RBAC patterns, and security configurations in Kubernetes clusters.

## NetworkPolicy Strategy

Start with default-deny, then allowlist required traffic. This is the only safe approach — permissive defaults leak.

| Policy Type | Purpose | Apply When |
|-------------|---------|------------|
| Default-deny all | Block all ingress/egress | Every namespace, always |
| Allow DNS egress | Enable service discovery | Every namespace with deny-all |
| Service-to-service | Specific pod communication | Per-service as needed |
| External egress | Internet access for specific pods | Only pods requiring external APIs |

### Default-Deny All Traffic

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
spec:
  podSelector: {}
  policyTypes: [Ingress, Egress]
```

### Allow DNS Egress (Required with deny-all)

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
spec:
  podSelector: {}
  policyTypes: [Egress]
  egress:
  - to:
    - namespaceSelector:
        matchLabels: { name: kube-system }
    ports:
    - { protocol: UDP, port: 53 }
```

### Service-to-Service Allowlist

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-api-to-db
spec:
  podSelector:
    matchLabels: { app: database }
  policyTypes: [Ingress]
  ingress:
  - from:
    - podSelector:
        matchLabels: { app: api-server }
    ports:
    - { protocol: TCP, port: 5432 }
```

### External Egress (Specific Pods Only)

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-external-api
spec:
  podSelector:
    matchLabels: { app: webhook-sender }
  policyTypes: [Egress]
  egress:
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
        except: [10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16]
    ports:
    - { protocol: TCP, port: 443 }
```

## RBAC Best Practices

| Principle | Implementation |
|-----------|---------------|
| Least privilege | Start with no permissions, add specific verbs/resources |
| Namespace-scoped | Use `Role` + `RoleBinding` (not `ClusterRole`) unless cross-namespace |
| No wildcard verbs | Never `verbs: ["*"]` — enumerate get, list, watch, create, update, delete |
| Service account per workload | One SA per deployment, not shared across services |
| Audit bindings | `kubectl auth can-i --list --as=system:serviceaccount:ns:sa` |

### Read-Only Role

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: readonly
  namespace: production
rules:
- apiGroups: [""]
  resources: [pods, services, configmaps]
  verbs: [get, list, watch]
- apiGroups: [apps]
  resources: [deployments, replicasets]
  verbs: [get, list, watch]
```

### Application Service Account

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: api-server
  namespace: production
automountServiceAccountToken: false  # Opt-in, not default
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: api-server-role
  namespace: production
rules:
- apiGroups: [""]
  resources: [configmaps]
  verbs: [get, watch]
  resourceNames: [api-config]  # Restrict to specific resources
- apiGroups: [""]
  resources: [secrets]
  verbs: [get]
  resourceNames: [api-secrets]
```

## Service Mesh mTLS (Istio)

```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
spec:
  mtls:
    mode: STRICT  # PERMISSIVE during migration
```

## Debugging RBAC

```bash
# Check what a service account can do
kubectl auth can-i list pods --as system:serviceaccount:ns:my-sa
kubectl auth can-i '*' '*' --as system:serviceaccount:ns:my-sa  # Check for admin

# Find all bindings for a user/SA
kubectl get rolebindings,clusterrolebindings -A -o wide | grep my-user

# List permissions for current context
kubectl auth can-i --list

# Check if NetworkPolicy is enforced (CNI-dependent)
kubectl get networkpolicy -A
kubectl describe networkpolicy default-deny-all -n production
```

## Gotchas

- **NetworkPolicy requires CNI support**: Calico, Cilium, Weave enforce NetworkPolicies. Default kubenet does NOT — policies are created but silently ignored.
- **DNS breaks with deny-all**: Always pair default-deny with DNS egress allow, or all service discovery fails.
- **RBAC wildcard drift**: `verbs: ["*"]` grants current AND future verbs. Enumerate explicitly.
- **automountServiceAccountToken**: Default is `true` — every pod gets a token. Set `false` on SA and opt-in per pod.
- **NetworkPolicy is additive**: Multiple policies on the same pod are OR'd for ingress/egress. You can't override a permissive policy with a restrictive one.
- **Sensitive values**: Validate that secrets are references (ESO, Vault paths), not plaintext. Regex-scan values files for high-entropy strings.

## Cross-References

- **devops:kubernetes-configuration** — pod security standards, security contexts, resource limits
- **security:secrets-management** — Vault, AWS Secrets Manager, External Secrets Operator
