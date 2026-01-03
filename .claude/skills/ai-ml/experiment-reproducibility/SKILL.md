---
name: experiment-reproducibility
description: Ensure ML experiment reproducibility through comprehensive seed management, environment locking, deterministic algorithms, and experiment tracking. Use when setting up research projects, debugging irreproducible results, or preparing code for publication.
---

# Experiment Reproducibility

Master the technical infrastructure for reproducible ML experiments, ensuring results can be reliably reproduced across different runs, machines, and time.

## When to Use This Skill

- Setting up new ML research projects
- Debugging results that vary between runs
- Preparing code for publication or release
- Sharing experiments with collaborators
- Meeting reproducibility requirements for conferences
- Building trusted experimental pipelines

## Comprehensive Seed Management

### Set All Random Seeds
```python
import os
import random
import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = True):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: The seed value to use
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        # CUDA deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # For CUDA >= 10.2, more determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # PyTorch deterministic algorithms (may raise errors for non-deterministic ops)
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Allow non-deterministic for speed
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_seed_state() -> dict:
    """Capture current random state for checkpointing."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_seed_state(state: dict):
    """Restore random state from checkpoint."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
```

### DataLoader Worker Seeds
```python
def worker_init_fn(worker_id: int):
    """Initialize each DataLoader worker with a unique seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Usage
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=worker_init_fn,
    generator=torch.Generator().manual_seed(42),  # For shuffle reproducibility
)
```

### Seed Management for Different Purposes
```python
from dataclasses import dataclass

@dataclass
class ExperimentSeeds:
    """Separate seeds for different experiment components."""
    data_split: int = 42      # Train/val/test splitting
    model_init: int = 43      # Weight initialization
    training: int = 44        # Training randomness (dropout, augmentation)
    evaluation: int = 45      # Evaluation sampling

    def apply_for_phase(self, phase: str):
        """Apply appropriate seed for experiment phase."""
        seeds = {
            'split': self.data_split,
            'init': self.model_init,
            'train': self.training,
            'eval': self.evaluation,
        }
        set_seed(seeds.get(phase, self.training))
```

## Environment Management

### Requirements Pinning
```bash
# Generate locked requirements with hashes
pip freeze > requirements.txt

# Or use pip-tools for better management
pip install pip-tools
pip-compile requirements.in --generate-hashes -o requirements.lock

# Install exact versions
pip install -r requirements.lock --require-hashes
```

### Conda Environment Export
```yaml
# environment.yml with exact versions
name: ml-research
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10.12
  - pytorch=2.1.0
  - torchvision=0.16.0
  - torchaudio=2.1.0
  - pytorch-cuda=12.1
  - numpy=1.24.3
  - pandas=2.0.3
  - scikit-learn=1.3.0
  - pip:
    - wandb==0.15.12
    - transformers==4.35.0
```

### Docker for Complete Reproducibility
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONHASHSEED=42
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.lock /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.lock

# Copy source code
COPY src/ /app/src/
COPY configs/ /app/configs/

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.train"]
```

## Configuration Management

### Hydra Configuration
```yaml
# configs/config.yaml
defaults:
  - model: resnet50
  - dataset: imagenet
  - optimizer: adamw
  - _self_

experiment:
  name: baseline
  seed: 42
  deterministic: true

training:
  epochs: 100
  batch_size: 256
  gradient_clip: 1.0

hydra:
  run:
    dir: outputs/${experiment.name}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: multirun/${experiment.name}
```

```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Log full config for reproducibility
    print(OmegaConf.to_yaml(cfg))

    # Save config alongside results
    OmegaConf.save(cfg, "config_used.yaml")

    # Apply seed
    set_seed(cfg.experiment.seed, cfg.experiment.deterministic)

    # Run experiment
    train(cfg)
```

### Pydantic Configuration
```python
from pydantic import BaseModel, Field
from typing import Optional
import json

class TrainingConfig(BaseModel):
    """Validated, serializable training configuration."""
    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Model
    model_name: str = "resnet50"
    pretrained: bool = True

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 0.01

    # Data
    data_path: str = "./data"
    num_workers: int = 4

    class Config:
        extra = "forbid"  # Prevent typos in config

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            return cls(**json.load(f))
```

## Code Version Tracking

### Git Information Logging
```python
import subprocess
from typing import Dict, Optional

def get_git_info() -> Dict[str, str]:
    """Capture git state for reproducibility."""
    def run_git(cmd: list) -> Optional[str]:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except subprocess.CalledProcessError:
            return None

    return {
        "commit_hash": run_git(["git", "rev-parse", "HEAD"]),
        "branch": run_git(["git", "branch", "--show-current"]),
        "is_dirty": run_git(["git", "status", "--porcelain"]) != "",
        "diff_stat": run_git(["git", "diff", "--stat"]),
        "remote_url": run_git(["git", "remote", "get-url", "origin"]),
    }


def log_experiment_context(save_dir: str):
    """Log all context needed for reproducibility."""
    import platform
    import torch
    import sys

    context = {
        "git": get_git_info(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    with open(f"{save_dir}/experiment_context.json", "w") as f:
        json.dump(context, f, indent=2)

    # Also save git diff for dirty repos
    if context["git"]["is_dirty"]:
        diff = subprocess.check_output(["git", "diff"]).decode()
        with open(f"{save_dir}/uncommitted_changes.diff", "w") as f:
            f.write(diff)
```

## Known Non-Determinism Sources

### PyTorch Non-Deterministic Operations
```python
# These operations may be non-deterministic:
# - torch.Tensor.index_add_()
# - torch.Tensor.scatter_add_()
# - torch.bincount()
# - torch.histc()
# - torch.nn.functional.interpolate() with certain modes
# - torch.nn.functional.grid_sample()
# - torch.nn.ReflectionPad2d()

# To identify non-deterministic operations:
torch.use_deterministic_algorithms(True)  # Will raise error on non-deterministic ops

# Workaround for specific operations
def deterministic_scatter_add(input, dim, index, src):
    """Deterministic alternative to scatter_add."""
    # Use a loop (slower but deterministic)
    output = input.clone()
    for i in range(src.size(0)):
        for j in range(src.size(1)):
            idx = index[i, j].item()
            output[i, idx] += src[i, j]
    return output
```

### Multi-Threading and Parallelism
```python
# Limit threads for reproducibility
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

# For DataLoader, use single worker for perfect reproducibility
# (or use worker_init_fn as shown above)
loader = DataLoader(dataset, num_workers=0)  # Single-threaded
```

## Experiment Tracking Integration

### Weights & Biases
```python
import wandb

def init_wandb_reproducible(config: dict, project: str):
    """Initialize W&B with full reproducibility context."""
    git_info = get_git_info()

    wandb.init(
        project=project,
        config={
            **config,
            "git_commit": git_info["commit_hash"],
            "git_branch": git_info["branch"],
            "git_dirty": git_info["is_dirty"],
        },
        settings=wandb.Settings(
            code_dir=".",  # Track code
            _save_requirements=True,  # Save requirements.txt
        )
    )

    # Log code
    wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    # Save config file
    wandb.save("config.yaml")
```

### MLflow
```python
import mlflow

def log_reproducibility_info():
    """Log reproducibility information to MLflow."""
    git_info = get_git_info()

    mlflow.log_param("git_commit", git_info["commit_hash"])
    mlflow.log_param("git_branch", git_info["branch"])
    mlflow.log_param("seed", 42)

    # Log environment
    mlflow.log_artifact("requirements.txt")
    mlflow.log_artifact("environment.yml")

    # Log config
    mlflow.log_artifact("config.yaml")
```

## Reproducibility Checklist

### Pre-Experiment
- [ ] Set all random seeds (Python, NumPy, PyTorch)
- [ ] Enable deterministic algorithms if needed
- [ ] Lock all dependency versions
- [ ] Commit all code changes
- [ ] Document configuration

### During Experiment
- [ ] Log git commit hash
- [ ] Save full configuration
- [ ] Track random states in checkpoints
- [ ] Log hardware/software environment

### Post-Experiment
- [ ] Verify results reproduce from checkpoint
- [ ] Test fresh environment installation
- [ ] Document any known non-determinism
- [ ] Archive complete experiment state

## REPRODUCE.md Template

```markdown
# Reproducing Results

## Environment Setup

```bash
# Clone repository
git clone https://github.com/user/repo.git
cd repo
git checkout <commit-hash>

# Create environment
conda env create -f environment.yml
conda activate ml-research

# Or with pip
pip install -r requirements.lock
```

## Data Preparation

```bash
# Download data
python scripts/download_data.py

# Preprocess (deterministic)
python scripts/preprocess.py --seed 42
```

## Training

```bash
# Reproduce main results (Table 1)
python train.py --config configs/main.yaml --seed 42

# Expected results:
# - Accuracy: 92.3% ± 0.2%
# - Training time: ~4 hours on A100
```

## Verification

```bash
# Verify checkpoint reproduces
python evaluate.py --checkpoint checkpoints/best.pt

# Expected output matches: results/expected_metrics.json
```

## Known Variations

- Results may vary by ±0.1% across different GPU architectures
- Multi-GPU training may have minor variations due to reduction order
```

## Best Practices

1. **Set seeds at the very start** before any imports that use randomness
2. **Use separate seeds** for data splitting vs training for cleaner experiments
3. **Document non-determinism** that you cannot eliminate
4. **Test reproducibility** by running the same experiment twice
5. **Version everything**: code, data, config, environment
6. **Checkpoint random states** along with model weights

## Common Pitfalls

- **Forgetting worker seeds**: DataLoader workers have independent random states
- **Import order**: Some libraries seed themselves on import
- **GPU non-determinism**: Some CUDA operations are inherently non-deterministic
- **Floating point**: Different hardware may give slightly different results
- **Implicit randomness**: Augmentation, dropout order, hash-based operations
