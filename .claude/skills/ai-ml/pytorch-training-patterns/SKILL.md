---
name: pytorch-training-patterns
description: Master PyTorch training patterns including custom training loops, gradient management, mixed precision, checkpointing, and optimization techniques. Use when implementing ML research, debugging training issues, or optimizing model training.
---

# PyTorch Training Patterns

Master professional PyTorch training patterns for ML research, from basic training loops to advanced optimization techniques.

## When to Use This Skill

- Implementing custom training loops for research
- Debugging training issues (NaN losses, gradient problems)
- Optimizing training performance and memory usage
- Implementing mixed precision training
- Managing model checkpoints and resuming training
- Building reusable training infrastructure

## Core Training Loop

### Basic Training Loop
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cuda",
    scheduler: Optional[Any] = None,
    clip_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    """Standard training epoch with best practices."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        # Move data to device
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but recommended)
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        # Optimizer step
        optimizer.step()

        # Scheduler step (if per-batch)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return {"loss": total_loss / num_batches}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluation loop with no gradients."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        predictions = outputs.argmax(dim=-1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total
    }
```

## Mixed Precision Training

### Automatic Mixed Precision (AMP)
```python
from torch.cuda.amp import autocast, GradScaler

def train_with_amp(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: str = "cuda",
) -> Dict[str, float]:
    """Training with automatic mixed precision."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Scaled backward pass
        scaler.scale(loss).backward()

        # Unscale gradients for clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return {"loss": total_loss / len(dataloader)}


# Usage
scaler = GradScaler()
for epoch in range(num_epochs):
    metrics = train_with_amp(model, train_loader, optimizer, criterion, scaler)
```

### BFloat16 Training (Modern GPUs)
```python
# For Ampere+ GPUs (A100, H100), bfloat16 is often better
with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# bfloat16 doesn't need loss scaling
loss.backward()
optimizer.step()
```

## Gradient Accumulation

```python
def train_with_accumulation(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    accumulation_steps: int = 4,
    device: str = "cuda",
) -> Dict[str, float]:
    """Training with gradient accumulation for larger effective batch sizes."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()

        total_loss += loss.item() * accumulation_steps

        # Update weights every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # Handle remaining gradients
    if (step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {"loss": total_loss / len(dataloader)}
```

## Checkpointing

### Save and Load Checkpoints
```python
from pathlib import Path
from typing import Optional
import json

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
    config: Optional[Dict] = None,
):
    """Save complete training state."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    # Save checkpoint
    torch.save(checkpoint, path)

    # Save metrics separately for easy access
    metrics_path = Path(path).with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump({"epoch": epoch, "metrics": metrics}, f, indent=2)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Load training state from checkpoint."""
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
    }
```

### Best Model Tracking
```python
class BestModelTracker:
    """Track and save best model during training."""

    def __init__(
        self,
        save_dir: str,
        metric_name: str = "val_loss",
        mode: str = "min",  # "min" or "max"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1

    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> bool:
        """Update best model if current is better."""
        current_value = metrics.get(self.metric_name)
        if current_value is None:
            return False

        is_better = (
            (self.mode == "min" and current_value < self.best_value) or
            (self.mode == "max" and current_value > self.best_value)
        )

        if is_better:
            self.best_value = current_value
            self.best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, metrics,
                str(self.save_dir / "best_model.pt"),
                **kwargs
            )
            return True

        return False
```

## Learning Rate Scheduling

### Common Schedulers
```python
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    LambdaLR,
)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# Cosine with warm restarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# One cycle (good for training from scratch)
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,  # Warmup percentage
)


# Custom warmup + cosine decay
def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Linear warmup followed by cosine decay."""
    import math

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
```

## Optimizer Patterns

### AdamW with Weight Decay Fix
```python
def configure_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Configure AdamW with proper weight decay handling."""
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't apply weight decay to biases and LayerNorm
        if "bias" in name or "LayerNorm" in name or "layernorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(optimizer_groups, lr=learning_rate)
```

### Layer-wise Learning Rate Decay (LLRD)
```python
def get_llrd_optimizer(
    model: nn.Module,
    base_lr: float = 1e-4,
    layer_decay: float = 0.9,
    weight_decay: float = 0.01,
):
    """Layer-wise learning rate decay for fine-tuning."""
    param_groups = []
    num_layers = len(list(model.children()))

    for layer_idx, (name, param) in enumerate(model.named_parameters()):
        if not param.requires_grad:
            continue

        # Calculate layer-specific learning rate
        lr = base_lr * (layer_decay ** (num_layers - layer_idx - 1))

        # Weight decay handling
        wd = 0.0 if "bias" in name or "norm" in name.lower() else weight_decay

        param_groups.append({
            "params": [param],
            "lr": lr,
            "weight_decay": wd,
        })

    return torch.optim.AdamW(param_groups)
```

## Memory Optimization

### Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            # Checkpoint each layer to trade compute for memory
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

### Memory-Efficient Attention
```python
# Use PyTorch 2.0+ scaled_dot_product_attention
import torch.nn.functional as F

def memory_efficient_attention(q, k, v, is_causal=False):
    """Use Flash Attention when available."""
    return F.scaled_dot_product_attention(
        q, k, v,
        is_causal=is_causal,
        # Automatically uses Flash Attention if available
    )
```

## Complete Training Script Template

```python
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.model = self._build_model().to(self.device)
        self.optimizer = configure_optimizer(self.model, config.lr, config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler() if config.use_amp else None

        # Scheduler
        total_steps = config.epochs * len(self.train_loader)
        self.scheduler = get_warmup_cosine_scheduler(
            self.optimizer, config.warmup_steps, total_steps
        )

        # Tracking
        self.best_tracker = BestModelTracker(config.save_dir, "val_loss", "min")
        self.start_epoch = 0

        # Resume if checkpoint exists
        if config.resume and Path(config.resume).exists():
            self._load_checkpoint(config.resume)

    def train(self):
        for epoch in range(self.start_epoch, self.config.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")

            # Training
            train_metrics = self._train_epoch()
            logger.info(f"Train: {train_metrics}")

            # Validation
            val_metrics = self._evaluate()
            logger.info(f"Val: {val_metrics}")

            # Save best model
            improved = self.best_tracker.update(
                self.model, self.optimizer, epoch,
                {"val_loss": val_metrics["loss"], **val_metrics},
                scheduler=self.scheduler,
                scaler=self.scaler,
            )

            if improved:
                logger.info(f"New best model! Val loss: {val_metrics['loss']:.4f}")

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    val_metrics,
                    f"{self.config.save_dir}/checkpoint_epoch_{epoch+1}.pt",
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                )

    def _train_epoch(self):
        if self.scaler:
            return train_with_amp(
                self.model, self.train_loader, self.optimizer,
                self.criterion, self.scaler, self.device
            )
        return train_epoch(
            self.model, self.train_loader, self.optimizer,
            self.criterion, self.device, self.scheduler
        )

    def _evaluate(self):
        return evaluate(self.model, self.val_loader, self.criterion, self.device)
```

## Best Practices

1. **Use `set_to_none=True`** in `optimizer.zero_grad()` for better performance
2. **Enable `non_blocking=True`** when moving data to GPU with pinned memory
3. **Use `torch.inference_mode()`** instead of `torch.no_grad()` for pure inference
4. **Compile models** with `torch.compile()` in PyTorch 2.0+ for speedup
5. **Profile regularly** using `torch.profiler` to find bottlenecks
6. **Use deterministic algorithms** when reproducibility matters

## Common Pitfalls

- **Forgetting model.train()/model.eval()**: Affects dropout and batch norm
- **Not moving all tensors to device**: Creates silent errors or crashes
- **Memory leaks**: Storing tensors in lists without `.detach()`
- **Wrong loss scaling**: Not dividing loss by accumulation steps
- **Scheduler timing**: Step per-batch vs per-epoch confusion
