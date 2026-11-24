set -euo pipefail
mkdir -p deploy docs
cat > deploy/checkout-api.yaml <<'YML'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkout-api
  labels:
    app: checkout-api
spec:
  replicas: 6
  selector:
    matchLabels:
      app: checkout-api
  template:
    metadata:
      labels:
        app: checkout-api
    spec:
      serviceAccountName: checkout-api
      containers:
        - name: api
          image: registry.example.com/checkout-api:2.14.0
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /live
              port: 8080
            periodSeconds: 10
            failureThreshold: 3
          resources:
            requests:
              cpu: "1"
              memory: 1Gi
            limits:
              cpu: "1"
              memory: 1Gi
YML
cat > docs/checkout-api.md <<'MD'
# checkout-api

- Synchronous and user-facing: the storefront waits on every call. The p99 latency budget is 80 ms.
- Measured per pod over the last 30 days: CPU averages 300m and peaks at 700m during sales;
  the memory working set averages 380Mi and peaks at 520Mi.
- The cluster runs other teams' workloads on the same nodes.
MD
