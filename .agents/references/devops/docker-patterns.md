# Docker Patterns

## Critical Rules

- **Never** run containers as root -- add `USER` directive with a non-root user in every Dockerfile
- **Never** pass secrets via `ARG` or `ENV` in Dockerfiles -- they persist in image layers; mount at runtime
- **Always** use `COPY` instead of `ADD` -- `ADD` auto-extracts tarballs and fetches URLs, creating attack surface
- **Always** pin base image versions and dependency versions -- unpinned tags drift and break builds silently
- **Always** scan images with `trivy` or equivalent before pushing to a registry

## Base Image Selection

| Workload | Base Image | Size | Notes |
|----------|-----------|------|-------|
| Python API / microservice | `python:3.11-slim` | ~150MB | Good default, glibc included |
| Minimal CLI tool | `python:3.11-alpine` | ~50MB | Missing glibc; breaks numpy, pandas |
| ML training (GPU) | `nvidia/cuda:12.2.0-runtime-ubuntu22.04` | ~3.5GB | Pair with pip, not conda |
| ML inference (GPU) | `nvidia/cuda:12.2.0-base-ubuntu22.04` | ~800MB | Smaller; runtime libs only |
| ML inference (CPU) | `python:3.11-slim` | ~150MB | No CUDA overhead |
| Distroless prod deploy | `gcr.io/distroless/python3` | ~50MB | No shell, no package manager |

## Optimized Python Dockerfile

### Multi-Stage ML Build

Separates build-time deps (compilers, headers) from runtime.

```dockerfile
# --- Build stage ---
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y python3-pip python3-dev
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runtime stage ---
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . /app
WORKDIR /app

CMD ["python3", "train.py"]
```

Result: devel image (~5GB) stays in build cache only. Runtime image drops ~3GB of compiler toolchain.

## Docker Compose for Dev Environments

### GPU Passthrough with Services

```yaml
version: "3.8"

services:
  train:
    build:
      context: .
      dockerfile: Dockerfile.train
    volumes:
      - ./j-data:/app/data
      - ./checkpoints:/app/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
      - NCCL_DEBUG=INFO
    shm_size: "8g"

  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    depends_on:
      - redis
    volumes:
      - ./src:/app/src

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_PASSWORD: dev
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

Key: `shm_size: "8g"` is required for PyTorch DataLoader with `num_workers > 0`. Default 64MB causes bus errors.

## .dockerignore Patterns

```
# Version control
.git
.gitignore

# Python
__pycache__
*.pyc
*.egg-info
.venv
venv

# ML artifacts -- never ship these in images
*.pt
*.pth
*.onnx
*.safetensors
checkpoints/
wandb/
mlruns/

# Dev tooling
.vscode
.idea
*.md
Dockerfile*
docker-compose*
.dockerignore
```

## Gotchas and Anti-Patterns

### Layer Cache Invalidation

**Problem**: Changing any file in `COPY . /app` busts cache for that layer and everything after it.

**Fix**: Copy dependency manifests first, install, then copy app code (shown above).

### pip Install Ordering

**Problem**: `pip install torch transformers datasets` in one line means changing `datasets` version reinstalls `torch` (2GB+).

**Fix**: Split into tiers:
```dockerfile
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
```

### CUDA Version Compatibility

CUDA runtime in image must match or be older than host driver's supported CUDA version. Check with:
```bash
nvidia-smi  # shows "CUDA Version: 12.x" -- this is the MAX supported
```

| Host Driver | Max CUDA Toolkit |
|------------|-----------------|
| 535.x | 12.2 |
| 545.x | 12.3 |
| 550.x | 12.4 |

Mismatch produces: `CUDA error: no kernel image is available for execution on the device`.

### Conda Bloat

Conda installs create massive images (often 8-12GB). Prefer pip with `--no-cache-dir`. If conda is unavoidable:
```dockerfile
RUN conda install --yes package && conda clean --all --force-pkgs-dirs
```

### Multi-Platform Builds

```bash
# Build for both amd64 and arm64
docker buildx build --platform linux/amd64,linux/arm64 -t myapp:latest --push .
```

Note: GPU images are amd64-only. Do not attempt multi-platform for CUDA-based images.

## Cross-References

- **devops:pipeline-design** -- CI/CD pipeline structure, deployment strategies, security scanning
- **devops:github-actions-patterns** -- container build actions, GHA cache for Docker layers
- **devops:gitops-workflow** -- ArgoCD/Flux image update automation, GitOps deployment
