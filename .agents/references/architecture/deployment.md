# Deployment Patterns

Deployment strategies, CI/CD pipelines, health checks, secret management, and environment promotion.

## Deployment Strategies

| Strategy | Downtime | Rollback | Best For | Risk |
|----------|----------|----------|----------|------|
| **Recreate** | Yes (~seconds) | Manual re-deploy | Dev/test, stateless services | Service unavailable during update |
| **Rolling** | No | Automatic (keep N old replicas) | Production, gradual updates | Old + new code coexist (test compatibility) |
| **Blue-Green** | No | Instant (switch load balancer) | High-traffic, critical services | Double resource usage during swap |
| **Canary** | No | Automatic (route to stable) | Risky releases (test % traffic) | Monitoring required, complexity |

```bash
# Rolling: gradually update replicas (k8s example)
kubectl set image deployment/app app=image:v2 --record
kubectl rollout status deployment/app

# Blue-green: switch traffic after validation
kubectl apply -f app-v2.yaml
kubectl patch svc app -p '{"spec":{"selector":{"version":"v2"}}}'

# Canary: route 5% to new version, monitor
kubectl apply -f app-canary.yaml
kubectl patch virtual-service app --type merge -p '{"spec":{"hosts":[...weights: [95,5]...}]}}'
```

## CI/CD Pipeline Stages

```yaml
# GitHub Actions / GitLab CI example
stages:
  - build:      # Compile, test, lint
  - unit-test:  # Fast validation
  - integration: # Database, external service tests
  - staging:    # Deploy to staging, smoke tests
  - approval:   # Manual review gate
  - production: # Deploy to prod, run smoke tests
  - monitor:    # Verify health checks, error rates
```

**Key rule**: Each stage should be ~5-10 min; fail fast.

## Health Checks (Kubernetes Probes)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app
spec:
  containers:
  - name: app
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 10
      timeoutSeconds: 3
      failureThreshold: 3
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5
    startupProbe:
      httpGet:
        path: /health/startup
        port: 8080
      initialDelaySeconds: 0
      periodSeconds: 10
      failureThreshold: 30
```

### Health Check Endpoints

```python
# Liveness: Is the app running? (restart if false)
@app.get("/health/live")
def liveness():
    # Lightweight: just return 200
    return {"status": "alive"}

# Readiness: Can the app accept traffic? (remove from LB if false)
@app.get("/health/ready")
def readiness():
    # Check dependencies: DB, cache, external APIs
    if not db.connected():
        return {"status": "not_ready"}, 503
    return {"status": "ready"}

# Startup: Has the app finished initialization? (give it time)
@app.get("/health/startup")
def startup():
    # Check warm caches, DB migrations, etc.
    if not app.initialized:
        return {"status": "starting"}, 503
    return {"status": "started"}
```

**Gotcha**: Make readiness checks strict; if they fail, traffic is removed (good). Liveness failures cause restarts (risky if thrashing).

## Rollback Strategies

```bash
# 1. Instant rollback (traffic switch, no code change)
kubectl rollout undo deployment/app

# 2. Revert commit + redeploy
git revert <commit-hash>
git push
# CI/CD auto-deploys

# 3. Database rollback (if schema changed)
# Keep migration reversible (DOWN scripts)
flyway info  # show applied migrations
# Revert schema BEFORE code rollback

# 4. Feature flags (no re-deploy needed)
# Kill feature, keep code in place
feature_flags.set("new_checkout", enabled=False)
```

## Environment Promotion (dev → staging → prod)

```yaml
# Single pipeline, different config per environment
environments:
  dev:
    database: dev.db
    replicas: 1
    resources: { cpu: 100m, memory: 256Mi }
    alerts: disabled
  staging:
    database: staging.db
    replicas: 2
    resources: { cpu: 500m, memory: 1Gi }
    alerts: warning level only
  production:
    database: prod.db
    replicas: 3+
    resources: { cpu: 2000m, memory: 4Gi }
    alerts: critical threshold

# Deploy: approve manually between stages
pipeline:
  - build (all)
  - test (all)
  - deploy-dev (auto)
  - deploy-staging (manual approval)
  - deploy-prod (manual approval + health checks)
```

## Secret Management

```python
# WRONG: Hardcode secrets
DATABASE_URL = "postgresql://user:password@host/db"

# RIGHT: Load from environment/secrets manager
import os
from dotenv import load_dotenv

load_dotenv(".env.local")  # local dev only
DATABASE_URL = os.getenv("DATABASE_URL")

# In production: use secrets manager
# K8s Secrets (encrypted at rest)
# AWS Secrets Manager / Parameter Store
# HashiCorp Vault
# 1Password CLI
```

```bash
# Kubernetes secrets
kubectl create secret generic db-secret \
  --from-literal=DATABASE_URL=postgresql://...

# Mount as environment variable
env:
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: db-secret
        key: DATABASE_URL
```

**Rule**: Never commit `.env`; use `.env.example` with placeholders.

## Gotchas

- **Blue-green waste**: Running two full environments costs 2x; watch costs.
- **Health check cascade**: Readiness failure can cause graceful shutdown; liveness failure restarts (can thrash).
- **Canary metrics**: Need real monitoring; "no errors in 1 min" is too short; use 5-10 min windows.
- **Rolling + stateful**: Ensure old and new versions can coexist (e.g., DB schema backwards-compatible).
- **Secrets in logs**: Sanitize error messages; never log DATABASE_URL or API keys.

---

## MCP Server Deployment

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "server.py"]
```

```python
# server.py -- use streamable HTTP for containerized deployment
mcp = FastMCP("my-server")
# ... register tools, resources ...
mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
```

### systemd (Linux)

```ini
[Unit]
Description=MCP Server
After=network.target

[Service]
Type=simple
User=mcp
WorkingDirectory=/opt/mcp-server
ExecStart=/opt/mcp-server/.venv/bin/python server.py
Restart=on-failure
RestartSec=5
Environment=DATABASE_URL=postgresql://localhost/mydb

[Install]
WantedBy=multi-user.target
```

### Authentication for Remote Servers

```python
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware

@mcp.tool()
def protected_tool(query: str) -> str:
    """This tool requires authentication."""
    ...

# Add auth middleware for HTTP transports
# MCP spec recommends OAuth 2.0 for remote servers
# Validate Bearer tokens on each request
```
