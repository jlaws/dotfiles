# Experiment Tracking Setup

Set up comprehensive ML experiment tracking infrastructure with W&B, MLflow, or Neptune for reproducible research.

## Requirements

- **Project context**: $ARGUMENTS (provide project name, framework (PyTorch/JAX/TensorFlow), and preferred tracking platform)

## Instructions

### 1. Analyze Project Requirements

Assess the project's needs:
- Framework being used (PyTorch, JAX, TensorFlow, scikit-learn)
- Scale of experiments (single GPU, multi-GPU, distributed)
- Team collaboration requirements
- Storage and cost constraints
- Integration with existing tools (Hydra, Lightning, etc.)

### 2. Platform Selection Guide

**Weights & Biases (wandb)**
- Best for: Team collaboration, experiment comparison, report generation
- Pros: Excellent UI, automatic logging, sweeps, artifacts
- Cons: Cloud-based (privacy concerns), cost at scale

**MLflow**
- Best for: Self-hosted, model registry needs, ML lifecycle management
- Pros: Open source, model versioning, deployment integration
- Cons: More setup required, less polished UI

**Neptune.ai**
- Best for: Large-scale experiments, metadata-heavy tracking
- Pros: Great for hyperparameter sweeps, flexible metadata
- Cons: Cloud-based, pricing at scale

### 3. Implementation Template

Provide a complete tracking setup:

```python
# config.py - Centralized experiment configuration
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os

@dataclass
class ExperimentConfig:
    """Experiment configuration with tracking metadata."""
    # Experiment identification
    project_name: str = "my-research"
    experiment_name: str = "baseline"
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    # Model configuration
    model_name: str = "resnet50"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Training configuration
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_steps: int = 1000

    # Data configuration
    dataset: str = "imagenet"
    data_path: str = "./data"
    num_workers: int = 4
    augmentation: str = "standard"

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Tracking
    log_interval: int = 100
    save_interval: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}
```

```python
# tracking.py - Unified tracking interface
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import json

class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking."""

    @abstractmethod
    def init(self, config: Dict[str, Any]) -> None:
        """Initialize tracking run."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics at a given step."""
        pass

    @abstractmethod
    def log_artifact(self, path: str, name: str, type: str) -> None:
        """Log an artifact (model, data, etc.)."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finalize the tracking run."""
        pass


class WandbTracker(ExperimentTracker):
    """Weights & Biases tracker implementation."""

    def __init__(self, project: str, entity: Optional[str] = None):
        import wandb
        self.wandb = wandb
        self.project = project
        self.entity = entity
        self.run = None

    def init(self, config: Dict[str, Any]) -> None:
        # Log git information
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode().strip()
            git_branch = subprocess.check_output(
                ["git", "branch", "--show-current"]
            ).decode().strip()
            git_diff = subprocess.check_output(
                ["git", "diff", "--stat"]
            ).decode().strip()
        except subprocess.CalledProcessError:
            git_hash = git_branch = git_diff = "N/A"

        self.run = self.wandb.init(
            project=self.project,
            entity=self.entity,
            config={
                **config,
                "git_hash": git_hash,
                "git_branch": git_branch,
                "git_diff": git_diff,
            },
            tags=config.get("tags", []),
            notes=config.get("notes", ""),
        )

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self.wandb.log(metrics, step=step)

    def log_artifact(self, path: str, name: str, type: str) -> None:
        artifact = self.wandb.Artifact(name=name, type=type)
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        self.wandb.finish()


class MLflowTracker(ExperimentTracker):
    """MLflow tracker implementation."""

    def __init__(self, tracking_uri: str, experiment_name: str):
        import mlflow
        self.mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = None

    def init(self, config: Dict[str, Any]) -> None:
        self.run = self.mlflow.start_run()
        self.mlflow.log_params(self._flatten_dict(config))

    def _flatten_dict(self, d: Dict, parent_key: str = '') -> Dict:
        """Flatten nested dict for MLflow params."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, name: str, type: str) -> None:
        self.mlflow.log_artifact(path)

    def finish(self) -> None:
        self.mlflow.end_run()
```

```python
# train.py - Example training loop with tracking
import torch
from pathlib import Path

def train(config: ExperimentConfig, tracker: ExperimentTracker):
    """Training loop with comprehensive tracking."""

    # Initialize tracking
    tracker.init(config.to_dict())

    # Set seeds for reproducibility
    set_seed(config.seed, config.deterministic)

    # Setup model, optimizer, data
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    train_loader, val_loader = create_dataloaders(config)

    # Track model architecture
    tracker.log_artifact(
        save_model_summary(model),
        name="model_architecture",
        type="model"
    )

    global_step = 0
    best_val_metric = float('inf')

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = compute_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # Log training metrics
            if global_step % config.log_interval == 0:
                tracker.log_metrics({
                    "train/loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }, step=global_step)

        # Validation
        val_metrics = evaluate(model, val_loader)
        tracker.log_metrics({
            f"val/{k}": v for k, v in val_metrics.items()
        }, step=global_step)

        # Save best model
        if val_metrics['loss'] < best_val_metric:
            best_val_metric = val_metrics['loss']
            checkpoint_path = save_checkpoint(model, optimizer, epoch, config)
            tracker.log_artifact(checkpoint_path, "best_model", "model")

    tracker.finish()


def set_seed(seed: int, deterministic: bool = True):
    """Set all seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### 4. Integration Patterns

**With Hydra:**
```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    tracker = WandbTracker(project=cfg.project.name)
    tracker.init(OmegaConf.to_container(cfg, resolve=True))
    train(cfg, tracker)
```

**With PyTorch Lightning:**
```python
from pytorch_lightning.loggers import WandbLogger

logger = WandbLogger(project="my-project", log_model=True)
trainer = Trainer(logger=logger)
```

### 5. What to Track

Ensure comprehensive logging:
- **Hyperparameters**: All configuration values
- **Metrics**: Train/val loss, accuracy, custom metrics per step
- **System**: GPU utilization, memory, training time
- **Code**: Git hash, diff, config files
- **Data**: Dataset version, preprocessing steps, splits
- **Artifacts**: Checkpoints, predictions, visualizations

## Output Format

Provide:
1. **Recommended platform** with justification
2. **Complete setup code** tailored to the project
3. **Configuration template** for experiments
4. **Integration instructions** for existing codebase
5. **Best practices checklist** for the team
