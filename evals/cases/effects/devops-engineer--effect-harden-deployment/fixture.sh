set -euo pipefail
cat > deploy.yaml <<'YAML'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: shop
  labels:
    app: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
        - name: api
          image: shop/api:latest
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: 250m
              memory: 256Mi
            limits:
              cpu: 500m
          livenessProbe:
            httpGet:
              path: /healthz/full
              port: 8080
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /healthz/full
              port: 8080
            periodSeconds: 5
YAML
