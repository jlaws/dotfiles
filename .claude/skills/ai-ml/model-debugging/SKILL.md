---
name: model-debugging
description: Debug ML model training issues including gradient problems, loss anomalies, overfitting, and performance degradation. Use when training fails, loss behaves unexpectedly, or model performance is poor.
---

# Model Debugging

Systematic approaches to debugging ML model training, from gradient issues to performance problems.

## When to Use This Skill

- Loss becomes NaN or infinity during training
- Model fails to learn (loss doesn't decrease)
- Training is unstable (loss spikes, oscillations)
- Model overfits severely
- Validation performance is unexpectedly poor
- Model behaves differently in train vs eval mode

## Gradient Debugging

### Gradient Health Checks
```python
import torch
from typing import Dict, List, Tuple

def check_gradients(model: torch.nn.Module) -> Dict[str, any]:
    """Comprehensive gradient health check."""
    stats = {
        "total_params": 0,
        "params_with_grad": 0,
        "zero_grad_params": [],
        "nan_grad_params": [],
        "inf_grad_params": [],
        "large_grad_params": [],  # > 100
        "small_grad_params": [],  # < 1e-7
        "grad_norms": {},
    }

    for name, param in model.named_parameters():
        if param.requires_grad:
            stats["total_params"] += 1

            if param.grad is None:
                continue

            stats["params_with_grad"] += 1
            grad = param.grad

            # Check for problematic gradients
            grad_norm = grad.norm().item()
            stats["grad_norms"][name] = grad_norm

            if torch.isnan(grad).any():
                stats["nan_grad_params"].append(name)
            elif torch.isinf(grad).any():
                stats["inf_grad_params"].append(name)
            elif grad_norm == 0:
                stats["zero_grad_params"].append(name)
            elif grad_norm > 100:
                stats["large_grad_params"].append((name, grad_norm))
            elif grad_norm < 1e-7:
                stats["small_grad_params"].append((name, grad_norm))

    return stats


def diagnose_gradient_issues(stats: Dict) -> List[str]:
    """Diagnose gradient problems and suggest fixes."""
    issues = []

    if stats["nan_grad_params"]:
        issues.append(
            f"NaN gradients in: {stats['nan_grad_params'][:5]}... "
            "Consider: lower learning rate, gradient clipping, check for log(0) or div/0"
        )

    if stats["inf_grad_params"]:
        issues.append(
            f"Infinite gradients in: {stats['inf_grad_params'][:5]}... "
            "Consider: gradient clipping, mixed precision scaling"
        )

    no_grad_ratio = 1 - stats["params_with_grad"] / max(stats["total_params"], 1)
    if no_grad_ratio > 0.1:
        issues.append(
            f"{no_grad_ratio:.1%} params have no gradients. "
            "Check: detached tensors, frozen layers, dead ReLUs"
        )

    if stats["zero_grad_params"]:
        issues.append(
            f"Zero gradients in {len(stats['zero_grad_params'])} params. "
            "Possible dead neurons or disconnected graph"
        )

    if stats["large_grad_params"]:
        issues.append(
            f"Large gradients (>100): {stats['large_grad_params'][:3]}... "
            "Consider: gradient clipping, lower learning rate"
        )

    return issues
```

### Gradient Flow Visualization
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_gradient_flow(model: torch.nn.Module, save_path: str = None):
    """Plot gradient flow through layers."""
    ave_grads = []
    max_grads = []
    layers = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, label="max gradient")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, label="avg gradient")
    plt.hlines(0, 0, len(ave_grads), linewidth=1, color="k")
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.xlabel("Layers")
    plt.ylabel("Gradient magnitude")
    plt.title("Gradient flow")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
```

### Gradient Hooks for Debugging
```python
def register_gradient_hooks(model: torch.nn.Module):
    """Register hooks to monitor gradients during training."""
    gradient_stats = {}

    def make_hook(name):
        def hook(grad):
            gradient_stats[name] = {
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "min": grad.min().item(),
                "max": grad.max().item(),
                "norm": grad.norm().item(),
                "has_nan": torch.isnan(grad).any().item(),
            }
            return grad
        return hook

    handles = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            handle = param.register_hook(make_hook(name))
            handles.append(handle)

    return gradient_stats, handles
```

## Loss Debugging

### NaN/Inf Loss Detection
```python
def safe_loss_computation(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
    model: torch.nn.Module,
) -> Tuple[torch.Tensor, Dict]:
    """Compute loss with comprehensive debugging info."""
    debug_info = {}

    # Check inputs
    debug_info["output_stats"] = {
        "has_nan": torch.isnan(outputs).any().item(),
        "has_inf": torch.isinf(outputs).any().item(),
        "min": outputs.min().item(),
        "max": outputs.max().item(),
    }

    # Check for problematic values before loss
    if debug_info["output_stats"]["has_nan"]:
        # Find which layer produced NaN
        debug_info["nan_source"] = find_nan_source(model, outputs)

    # Compute loss
    loss = criterion(outputs, targets)

    debug_info["loss_value"] = loss.item() if not torch.isnan(loss) else "NaN"

    # If loss is bad, provide diagnosis
    if torch.isnan(loss) or torch.isinf(loss):
        debug_info["diagnosis"] = diagnose_bad_loss(outputs, targets, criterion)

    return loss, debug_info


def diagnose_bad_loss(outputs, targets, criterion):
    """Diagnose why loss became NaN/Inf."""
    diagnosis = []

    # Check for log(0) in cross entropy
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        probs = torch.softmax(outputs, dim=-1)
        min_prob = probs.min().item()
        if min_prob < 1e-7:
            diagnosis.append(f"Very small probabilities: {min_prob:.2e}")

    # Check for extreme values
    if outputs.abs().max() > 1e6:
        diagnosis.append(f"Extreme output values: {outputs.abs().max():.2e}")

    # Check target validity
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        num_classes = outputs.size(-1)
        invalid_targets = (targets < 0) | (targets >= num_classes)
        if invalid_targets.any():
            diagnosis.append(f"Invalid target indices found")

    return diagnosis
```

### Loss Landscape Analysis
```python
def compute_loss_landscape(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    direction1: Dict[str, torch.Tensor] = None,
    direction2: Dict[str, torch.Tensor] = None,
    steps: int = 21,
    range_val: float = 1.0,
    device: str = "cuda",
):
    """Compute 2D loss landscape around current parameters."""
    # Save original parameters
    original_params = {n: p.clone() for n, p in model.named_parameters()}

    # Generate random directions if not provided
    if direction1 is None:
        direction1 = {n: torch.randn_like(p) for n, p in model.named_parameters()}
        # Normalize
        norm1 = sum(d.norm()**2 for d in direction1.values()).sqrt()
        direction1 = {n: d / norm1 for n, d in direction1.items()}

    if direction2 is None:
        direction2 = {n: torch.randn_like(p) for n, p in model.named_parameters()}
        norm2 = sum(d.norm()**2 for d in direction2.values()).sqrt()
        direction2 = {n: d / norm2 for n, d in direction2.items()}

    # Compute losses
    alphas = np.linspace(-range_val, range_val, steps)
    betas = np.linspace(-range_val, range_val, steps)
    losses = np.zeros((steps, steps))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Perturb parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(
                        original_params[name] +
                        alpha * direction1[name] +
                        beta * direction2[name]
                    )

            # Compute loss
            loss = evaluate_loss(model, dataloader, criterion, device)
            losses[i, j] = loss

    # Restore original parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(original_params[name])

    return alphas, betas, losses
```

## Overfitting Diagnosis

### Overfitting Detection
```python
from collections import deque

class OverfittingDetector:
    """Detect overfitting during training."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.train_losses = deque(maxlen=patience)
        self.val_losses = deque(maxlen=patience)
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def update(self, train_loss: float, val_loss: float) -> Dict[str, any]:
        """Update with new epoch results and return diagnosis."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        diagnosis = {
            "is_overfitting": False,
            "gap": val_loss - train_loss,
            "gap_trend": None,
            "recommendation": None,
        }

        # Check for improvement
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # Diagnose overfitting
        if len(self.train_losses) >= 5:
            train_trend = np.polyfit(range(len(self.train_losses)),
                                     list(self.train_losses), 1)[0]
            val_trend = np.polyfit(range(len(self.val_losses)),
                                   list(self.val_losses), 1)[0]

            # Train decreasing but val increasing = overfitting
            if train_trend < -self.min_delta and val_trend > self.min_delta:
                diagnosis["is_overfitting"] = True
                diagnosis["gap_trend"] = "increasing"
                diagnosis["recommendation"] = (
                    "Overfitting detected. Try: "
                    "1) More regularization (dropout, weight decay) "
                    "2) Data augmentation "
                    "3) Early stopping "
                    "4) Reduce model capacity"
                )

            # Both stuck = underfitting
            elif abs(train_trend) < self.min_delta and abs(val_trend) < self.min_delta:
                if train_loss > 0.5:  # Threshold depends on task
                    diagnosis["recommendation"] = (
                        "Training seems stuck. Try: "
                        "1) Increase learning rate "
                        "2) Increase model capacity "
                        "3) Check data loading"
                    )

        return diagnosis
```

### Layer-wise Analysis
```python
def analyze_layer_statistics(model: torch.nn.Module) -> Dict[str, Dict]:
    """Analyze statistics of each layer's weights and activations."""
    stats = {}

    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            weight = module.weight.data
            stats[name] = {
                "weight_mean": weight.mean().item(),
                "weight_std": weight.std().item(),
                "weight_min": weight.min().item(),
                "weight_max": weight.max().item(),
                "weight_norm": weight.norm().item(),
            }

            if hasattr(module, 'bias') and module.bias is not None:
                bias = module.bias.data
                stats[name].update({
                    "bias_mean": bias.mean().item(),
                    "bias_std": bias.std().item(),
                })

            # Check for dead neurons (for ReLU-based networks)
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                # Weights with very small magnitude
                small_weights = (weight.abs() < 1e-6).float().mean().item()
                stats[name]["small_weight_ratio"] = small_weights

    return stats


def find_dead_neurons(model: torch.nn.Module, dataloader, device="cuda"):
    """Find neurons that never activate (dead ReLUs)."""
    activation_counts = {}

    def make_hook(name):
        def hook(module, input, output):
            # Count non-zero activations
            active = (output > 0).float().mean(dim=0)  # Per neuron
            if name not in activation_counts:
                activation_counts[name] = active
            else:
                activation_counts[name] += active
        return hook

    # Register hooks on ReLU layers
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.GELU)):
            handles.append(module.register_forward_hook(make_hook(name)))

    # Run through data
    model.eval()
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            model(inputs)
            num_batches += 1
            if num_batches >= 100:
                break

    # Clean up hooks
    for handle in handles:
        handle.remove()

    # Find dead neurons
    dead_neurons = {}
    for name, counts in activation_counts.items():
        counts = counts / num_batches
        dead_ratio = (counts < 0.01).float().mean().item()
        if dead_ratio > 0:
            dead_neurons[name] = {
                "dead_ratio": dead_ratio,
                "total_neurons": counts.numel(),
            }

    return dead_neurons
```

## Training Instability

### Learning Rate Finder
```python
class LRFinder:
    """Find optimal learning rate range."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: str = "cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Save initial state
        self.model_state = model.state_dict()
        self.optimizer_state = optimizer.state_dict()

    def find(
        self,
        dataloader: torch.utils.data.DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        smooth_f: float = 0.05,
    ) -> Tuple[List[float], List[float]]:
        """Run learning rate range test."""
        # Calculate lr multiplier per step
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)

        lrs = []
        losses = []
        best_loss = float('inf')
        avg_loss = 0.0

        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr

        self.model.train()
        iterator = iter(dataloader)

        for i in range(num_iter):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Check for explosion
            if torch.isnan(loss) or loss.item() > 4 * best_loss:
                break

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Record
            current_lr = self.optimizer.param_groups[0]['lr']
            lrs.append(current_lr)

            # Smooth loss
            if i == 0:
                avg_loss = loss.item()
            else:
                avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss
            losses.append(avg_loss)

            best_loss = min(best_loss, avg_loss)

            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_mult

        # Restore initial state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        return lrs, losses

    def plot(self, lrs: List[float], losses: List[float], save_path: str = None):
        """Plot learning rate vs loss."""
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")

        # Find suggested LR (steepest descent)
        gradients = np.gradient(losses)
        min_grad_idx = np.argmin(gradients)
        suggested_lr = lrs[min_grad_idx]
        plt.axvline(x=suggested_lr, color='r', linestyle='--',
                    label=f'Suggested LR: {suggested_lr:.2e}')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        plt.show()

        return suggested_lr
```

## Debugging Checklist

### Training Not Starting
- [ ] Check data loading (verify batch shapes, values)
- [ ] Verify model output shapes match targets
- [ ] Check loss function compatibility
- [ ] Verify model is on correct device
- [ ] Check for frozen parameters

### Loss Not Decreasing
- [ ] Learning rate too low or too high
- [ ] Check gradient flow (vanishing/exploding)
- [ ] Verify labels are correct
- [ ] Check for data leakage (train on test)
- [ ] Model capacity too low

### NaN Loss
- [ ] Lower learning rate
- [ ] Add gradient clipping
- [ ] Check for log(0) or division by zero
- [ ] Verify input normalization
- [ ] Check for exploding activations

### Overfitting
- [ ] Add dropout/regularization
- [ ] Use data augmentation
- [ ] Reduce model size
- [ ] Get more training data
- [ ] Use early stopping

## Best Practices

1. **Log everything**: Loss, gradients, learning rate, metrics per epoch
2. **Start simple**: Overfit on small batch first
3. **Use hooks**: Register gradient hooks for debugging
4. **Visualize**: Plot losses, gradients, activations
5. **Compare baselines**: Verify your implementation against known good results

## Common Pitfalls

- **Silent data type issues**: Float16 overflow, integer division
- **Forgotten model.eval()**: Dropout/batchnorm behave differently
- **Memory leaks**: Storing tensors without detaching
- **Data augmentation bugs**: Augmenting validation data
- **Shuffled validation**: Non-deterministic validation metrics
