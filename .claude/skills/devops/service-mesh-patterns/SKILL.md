---
name: service-mesh-patterns
description: Configure service mesh traffic management with Istio and Linkerd including routing, load balancing, circuit breakers, canary deployments, and zero-trust networking. Use when implementing service mesh policies, progressive delivery, or resilience patterns.
---

# Service Mesh Patterns

Comprehensive guide to service mesh traffic management for both Istio and Linkerd deployments.

## When to Use This Skill

- Configuring service-to-service routing
- Implementing canary or blue-green deployments
- Setting up circuit breakers and retries
- Load balancing configuration
- Traffic mirroring for testing
- Fault injection for chaos engineering
- Implementing automatic mTLS
- Multi-cluster service mesh

## Comparison: Istio vs Linkerd

| Feature | Istio | Linkerd |
|---------|-------|---------|
| **Complexity** | Feature-rich, more complex | Lightweight, simpler |
| **Resource Usage** | Higher memory/CPU | Lower footprint |
| **mTLS** | Configurable | Always-on by default |
| **Traffic Management** | VirtualService/DestinationRule | ServiceProfile/TrafficSplit |
| **Gateway** | Built-in Gateway resource | Uses external ingress |
| **Best For** | Complex enterprise scenarios | Simple, security-focused deployments |

---

## Part 1: Istio Traffic Management

### Core Resources

| Resource | Purpose | Scope |
|----------|---------|-------|
| **VirtualService** | Route traffic to destinations | Host-based |
| **DestinationRule** | Define policies after routing | Service-based |
| **Gateway** | Configure ingress/egress | Cluster edge |
| **ServiceEntry** | Add external services | Mesh-wide |

### Traffic Flow

```
Client → Gateway → VirtualService → DestinationRule → Service
                   (routing)        (policies)        (pods)
```

### Basic Routing

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews-route
spec:
  hosts:
    - reviews
  http:
    - match:
        - headers:
            end-user:
              exact: jason
      route:
        - destination:
            host: reviews
            subset: v2
    - route:
        - destination:
            host: reviews
            subset: v1
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: reviews-destination
spec:
  host: reviews
  subsets:
    - name: v1
      labels:
        version: v1
    - name: v2
      labels:
        version: v2
```

### Istio Canary Deployment

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: my-service-canary
spec:
  hosts:
    - my-service
  http:
    - route:
        - destination:
            host: my-service
            subset: stable
          weight: 90
        - destination:
            host: my-service
            subset: canary
          weight: 10
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: my-service-dr
spec:
  host: my-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        h2UpgradePolicy: UPGRADE
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
  subsets:
    - name: stable
      labels:
        version: stable
    - name: canary
      labels:
        version: canary
```

### Istio Circuit Breaker

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: circuit-breaker
spec:
  host: my-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
        maxRequestsPerConnection: 10
        maxRetries: 3
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 30
```

### Istio Retry and Timeout

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ratings-retry
spec:
  hosts:
    - ratings
  http:
    - route:
        - destination:
            host: ratings
      timeout: 10s
      retries:
        attempts: 3
        perTryTimeout: 3s
        retryOn: connect-failure,refused-stream,unavailable,cancelled,retriable-4xx,503
        retryRemoteLocalities: true
```

### Traffic Mirroring

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: mirror-traffic
spec:
  hosts:
    - my-service
  http:
    - route:
        - destination:
            host: my-service
            subset: v1
      mirror:
        host: my-service
        subset: v2
      mirrorPercentage:
        value: 100.0
```

### Fault Injection

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: fault-injection
spec:
  hosts:
    - ratings
  http:
    - fault:
        delay:
          percentage:
            value: 10
          fixedDelay: 5s
        abort:
          percentage:
            value: 5
          httpStatus: 503
      route:
        - destination:
            host: ratings
```

### Istio Ingress Gateway

```yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: my-gateway
spec:
  selector:
    istio: ingressgateway
  servers:
    - port:
        number: 443
        name: https
        protocol: HTTPS
      tls:
        mode: SIMPLE
        credentialName: my-tls-secret
      hosts:
        - "*.example.com"
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: my-vs
spec:
  hosts:
    - "api.example.com"
  gateways:
    - my-gateway
  http:
    - match:
        - uri:
            prefix: /api/v1
      route:
        - destination:
            host: api-service
            port:
              number: 8080
```

### Istio Load Balancing

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: load-balancing
spec:
  host: my-service
  trafficPolicy:
    loadBalancer:
      simple: ROUND_ROBIN  # or LEAST_CONN, RANDOM, PASSTHROUGH
---
# Consistent hashing for sticky sessions
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: sticky-sessions
spec:
  host: my-service
  trafficPolicy:
    loadBalancer:
      consistentHash:
        httpHeaderName: x-user-id
        # or: httpCookie, useSourceIp, httpQueryParameterName
```

### Istio Debugging

```bash
# Check VirtualService configuration
istioctl analyze

# View effective routes
istioctl proxy-config routes deploy/my-app -o json

# Check endpoint discovery
istioctl proxy-config endpoints deploy/my-app

# Debug traffic
istioctl proxy-config log deploy/my-app --level debug
```

---

## Part 2: Linkerd Patterns

### Architecture

```
┌─────────────────────────────────────────────┐
│                Control Plane                 │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ destiny │ │ identity │ │ proxy-inject │ │
│  └─────────┘ └──────────┘ └──────────────┘ │
└─────────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────┐
│                 Data Plane                   │
│  ┌─────┐    ┌─────┐    ┌─────┐             │
│  │proxy│────│proxy│────│proxy│             │
│  └─────┘    └─────┘    └─────┘             │
│     │           │           │               │
│  ┌──┴──┐    ┌──┴──┐    ┌──┴──┐            │
│  │ app │    │ app │    │ app │            │
│  └─────┘    └─────┘    └─────┘            │
└─────────────────────────────────────────────┘
```

### Key Resources

| Resource | Purpose |
|----------|---------|
| **ServiceProfile** | Per-route metrics, retries, timeouts |
| **TrafficSplit** | Canary deployments, A/B testing |
| **Server** | Define server-side policies |
| **ServerAuthorization** | Access control policies |

### Installation

```bash
# Install CLI
curl --proto '=https' --tlsv1.2 -sSfL https://run.linkerd.io/install | sh

# Validate cluster
linkerd check --pre

# Install CRDs
linkerd install --crds | kubectl apply -f -

# Install control plane
linkerd install | kubectl apply -f -

# Verify installation
linkerd check

# Install viz extension (optional)
linkerd viz install | kubectl apply -f -
```

### Inject Namespace

```yaml
# Automatic injection for namespace
apiVersion: v1
kind: Namespace
metadata:
  name: my-app
  annotations:
    linkerd.io/inject: enabled
```

### Service Profile with Retries

```yaml
apiVersion: linkerd.io/v1alpha2
kind: ServiceProfile
metadata:
  name: my-service.my-namespace.svc.cluster.local
  namespace: my-namespace
spec:
  routes:
    - name: GET /api/users
      condition:
        method: GET
        pathRegex: /api/users
      responseClasses:
        - condition:
            status:
              min: 500
              max: 599
          isFailure: true
      isRetryable: true
    - name: POST /api/users
      condition:
        method: POST
        pathRegex: /api/users
      isRetryable: false
    - name: GET /api/users/{id}
      condition:
        method: GET
        pathRegex: /api/users/[^/]+
      timeout: 5s
      isRetryable: true
  retryBudget:
    retryRatio: 0.2
    minRetriesPerSecond: 10
    ttl: 10s
```

### Linkerd Traffic Split (Canary)

```yaml
apiVersion: split.smi-spec.io/v1alpha1
kind: TrafficSplit
metadata:
  name: my-service-canary
  namespace: my-namespace
spec:
  service: my-service
  backends:
    - service: my-service-stable
      weight: 900m  # 90%
    - service: my-service-canary
      weight: 100m  # 10%
```

### Server Authorization Policy

```yaml
# Define the server
apiVersion: policy.linkerd.io/v1beta1
kind: Server
metadata:
  name: my-service-http
  namespace: my-namespace
spec:
  podSelector:
    matchLabels:
      app: my-service
  port: http
  proxyProtocol: HTTP/1
---
# Allow traffic from specific clients
apiVersion: policy.linkerd.io/v1beta1
kind: ServerAuthorization
metadata:
  name: allow-frontend
  namespace: my-namespace
spec:
  server:
    name: my-service-http
  client:
    meshTLS:
      serviceAccounts:
        - name: frontend
          namespace: my-namespace
---
# Allow unauthenticated traffic (e.g., from ingress)
apiVersion: policy.linkerd.io/v1beta1
kind: ServerAuthorization
metadata:
  name: allow-ingress
  namespace: my-namespace
spec:
  server:
    name: my-service-http
  client:
    unauthenticated: true
    networks:
      - cidr: 10.0.0.0/8
```

### HTTPRoute for Advanced Routing

```yaml
apiVersion: policy.linkerd.io/v1beta2
kind: HTTPRoute
metadata:
  name: my-route
  namespace: my-namespace
spec:
  parentRefs:
    - name: my-service
      kind: Service
      group: core
      port: 8080
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /api/v2
        - headers:
            - name: x-api-version
              value: v2
      backendRefs:
        - name: my-service-v2
          port: 8080
    - matches:
        - path:
            type: PathPrefix
            value: /api
      backendRefs:
        - name: my-service-v1
          port: 8080
```

### Multi-cluster Setup

```bash
# On each cluster, install with cluster credentials
linkerd multicluster install | kubectl apply -f -

# Link clusters
linkerd multicluster link --cluster-name west \
  --api-server-address https://west.example.com:6443 \
  | kubectl apply -f -

# Export a service to other clusters
kubectl label svc/my-service mirror.linkerd.io/exported=true

# Verify cross-cluster connectivity
linkerd multicluster check
linkerd multicluster gateways
```

### Linkerd Monitoring & Debugging

```bash
# Live traffic view
linkerd viz top deploy/my-app

# Per-route metrics
linkerd viz routes deploy/my-app

# Check proxy status
linkerd viz stat deploy -n my-namespace

# View service dependencies
linkerd viz edges deploy -n my-namespace

# Dashboard
linkerd viz dashboard

# Check injection status
linkerd check --proxy -n my-namespace

# View proxy logs
kubectl logs deploy/my-app -c linkerd-proxy

# Debug identity/TLS
linkerd identity -n my-namespace

# Tap traffic (live)
linkerd viz tap deploy/my-app --to deploy/my-backend
```

---

## Best Practices

### General
- **Start simple** - Add complexity incrementally
- **Set timeouts** - Always configure reasonable timeouts
- **Enable retries** - But with backoff and limits
- **Monitor golden metrics** - Success rate, latency, throughput

### Istio-Specific
- **Use subsets** - Version your services clearly
- **Enable outlier detection** - For circuit breaker functionality
- **Don't over-retry** - Can cause cascading failures
- **Don't mirror to production** - Mirror to test environments

### Linkerd-Specific
- **Enable mTLS everywhere** - It's automatic with Linkerd
- **Use ServiceProfiles** - Get per-route metrics and retries
- **Set retry budgets** - Prevent retry storms
- **Always run `linkerd check`** - After any changes

## Resources

### Istio
- [Istio Traffic Management](https://istio.io/latest/docs/concepts/traffic-management/)
- [Virtual Service Reference](https://istio.io/latest/docs/reference/config/networking/virtual-service/)
- [Destination Rule Reference](https://istio.io/latest/docs/reference/config/networking/destination-rule/)

### Linkerd
- [Linkerd Documentation](https://linkerd.io/2.14/overview/)
- [Service Profiles](https://linkerd.io/2.14/features/service-profiles/)
- [Authorization Policy](https://linkerd.io/2.14/features/server-policy/)

## Related Skills

- `observability-stack` - For metrics, dashboards, and mesh monitoring
- `distributed-tracing` - For request tracing across services
