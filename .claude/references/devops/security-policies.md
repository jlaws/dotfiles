# Security Policy YAML Examples

## Default-Deny NetworkPolicy

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
spec:
  podSelector: {}
  policyTypes: [Ingress, Egress]
```

## Allow DNS Egress

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
kubectl auth can-i list pods --as system:serviceaccount:ns:my-sa
kubectl auth can-i '*' '*' --as system:serviceaccount:ns:my-sa  # Check for admin
kubectl get rolebindings,clusterrolebindings -A -o wide | grep my-user
```

## Config Validation Gotchas

- **Drift between environments**: Use overlays/values files per env, never manual edits. Diff configs across envs in CI.
- **Hot-reload pitfalls**: Validate new config before applying. If validation fails in prod, keep current config and alert -- never crash.
- **Config migration**: Version your config schema. When schema changes, write explicit up/down migrations. Never silently ignore unknown fields.
- **Sensitive values**: Validate that secrets are references (ESO, Vault paths), not plaintext. Regex-scan values files for high-entropy strings.
