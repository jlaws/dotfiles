---
name: continual-and-online-learning
description: Implement continual learning systems that adapt to new data without catastrophic forgetting
---

# Continual and Online Learning

## Approach Decision Table

| Update Frequency | Data Volume | Forgetting Risk | Approach | Complexity |
|-----------------|-------------|-----------------|----------|------------|
| Monthly | Large batches | Moderate | Full fine-tune + EWC | Low |
| Weekly | Medium batches | High | Experience replay + EWC | Medium |
| Daily | Small batches | High | Online LoRA + validation gating | Medium |
| Real-time / streaming | Individual samples | Severe | Streaming EMA + drift detection | Medium |
| On-demand merge | N/A | Low | Model merging (Fisher-weighted) | Low |
| Periodic consolidation | Accumulated | Moderate | Linear interpolation / EMA merge | Low |

### When to Use What
- **EWC**: Known task boundaries, moderate parameter count, preserve specific capabilities.
- **Replay buffer**: No clear task boundaries, diverse data, can store examples.
- **Online LoRA**: Large base model, frequent small updates, need reversibility.
- **Model merging**: Multiple checkpoints, no training budget at merge time.

## Elastic Weight Consolidation (EWC)

```python
import torch
import torch.nn as nn

class EWCTrainer:
    """Penalize changes to parameters important for previous tasks."""

    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.saved_params: dict[str, torch.Tensor] = {}
        self.fisher_diag: dict[str, torch.Tensor] = {}

    def compute_fisher(self, dataloader, device="cuda", num_samples=500):
        """Diagonal Fisher information from task data."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                  if p.requires_grad}
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            inputs, targets = batch["input"].to(device), batch["target"].to(device)
            self.model.zero_grad()
            loss = nn.functional.cross_entropy(self.model(inputs), targets)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            count += inputs.size(0)
        for n in fisher:
            fisher[n] /= count
        self.fisher_diag = fisher
        self.saved_params = {n: p.data.clone() for n, p in self.model.named_parameters()
                             if p.requires_grad}

    def ewc_loss(self) -> torch.Tensor:
        """Quadratic penalty: F_i * (theta - theta_old)^2."""
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if n in self.fisher_diag:
                loss += (self.fisher_diag[n] * (p - self.saved_params[n]).pow(2)).sum()
        return self.ewc_lambda * loss

    def train_step(self, batch, criterion, optimizer, device="cuda"):
        """Training step with EWC regularization."""
        self.model.train()
        inputs, targets = batch["input"].to(device), batch["target"].to(device)
        task_loss = criterion(self.model(inputs), targets)
        total_loss = task_loss + self.ewc_loss()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        return {"task_loss": task_loss.item(), "ewc_loss": (total_loss - task_loss).item()}
```

## Experience Replay with Reservoir Sampling

```python
import random
import torch

class ReservoirReplayBuffer:
    """Reservoir sampling: uniform retention probability for any past example."""

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.buffer: list[dict] = []
        self.total_seen = 0

    def add(self, example: dict):
        self.total_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(example)
        else:
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = example

    def sample(self, batch_size: int) -> list[dict]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

def train_with_replay(model, dataloader, replay_buffer, optimizer,
                      criterion, replay_ratio=0.5, device="cuda"):
    """Interleave current task data with replay buffer samples."""
    model.train()
    for batch in dataloader:
        inputs, targets = batch["input"].to(device), batch["target"].to(device)
        new_loss = criterion(model(inputs), targets)

        replay_loss = torch.tensor(0.0, device=device)
        if len(replay_buffer.buffer) > 0:
            rb = replay_buffer.sample(inputs.size(0))
            r_in = torch.stack([r["input"] for r in rb]).to(device)
            r_tgt = torch.stack([r["target"] for r in rb]).to(device)
            replay_loss = criterion(model(r_in), r_tgt)

        total = (1 - replay_ratio) * new_loss + replay_ratio * replay_loss
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        for i in range(inputs.size(0)):
            replay_buffer.add({"input": inputs[i].cpu(), "target": targets[i].cpu()})
```

## Online LoRA with Validation Gating

```python
import torch
from peft import LoraConfig, get_peft_model

def setup_online_lora(base_model, r=16, alpha=32, targets=None):
    """LoRA for online updates: small, fast, reversible."""
    config = LoraConfig(r=r, lora_alpha=alpha,
                        target_modules=targets or ["q_proj", "v_proj"],
                        lora_dropout=0.05, bias="none")
    return get_peft_model(base_model, config)

def gated_update(model, new_data, val_data, optimizer, criterion,
                 max_epochs=3, threshold=0.02, device="cuda"):
    """Commit LoRA update only if validation doesn't regress."""
    snapshot = {n: p.data.clone() for n, p in model.named_parameters() if "lora" in n}
    baseline = _eval_score(model, val_data, criterion, device)

    model.train()
    for _ in range(max_epochs):
        for batch in new_data:
            inputs, targets = batch["input"].to(device), batch["target"].to(device)
            optimizer.zero_grad()
            criterion(model(inputs), targets).backward()
            optimizer.step()

    new_score = _eval_score(model, val_data, criterion, device)
    if baseline - new_score > threshold:  # regression detected
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in snapshot:
                    p.data.copy_(snapshot[n])
        return {"accepted": False, "regression": baseline - new_score}
    return {"accepted": True, "improvement": new_score - baseline}

def _eval_score(model, dataloader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for b in dataloader:
            x, y = b["input"].to(device), b["target"].to(device)
            total += criterion(model(x), y).item() * x.size(0)
            count += x.size(0)
    return -total / count  # negative loss (higher = better)
```

## Model Merging

```python
import torch
from copy import deepcopy

def linear_interpolation(model_a, model_b, alpha=0.5):
    """theta = alpha * A + (1-alpha) * B."""
    merged = deepcopy(model_a)
    with torch.no_grad():
        for (_, pm), (_, pa), (_, pb) in zip(
            merged.named_parameters(), model_a.named_parameters(),
            model_b.named_parameters()):
            pm.data = alpha * pa.data + (1 - alpha) * pb.data
    return merged

def ema_update(online_model, target_model, decay=0.999):
    """EMA: target = decay * target + (1 - decay) * online."""
    with torch.no_grad():
        for po, pt in zip(online_model.parameters(), target_model.parameters()):
            pt.data = decay * pt.data + (1 - decay) * po.data

def fisher_weighted_merge(model_a, model_b, fisher_a, fisher_b):
    """Importance-aware merge weighted by Fisher information."""
    merged = deepcopy(model_a)
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())
    with torch.no_grad():
        for n, pm in merged.named_parameters():
            if n in fisher_a and n in fisher_b:
                fa, fb = fisher_a[n], fisher_b[n]
                pm.data = (fa * params_a[n].data + fb * params_b[n].data) / (fa + fb + 1e-8)
    return merged
```

## Streaming Drift Detection

```python
import numpy as np
from collections import deque

class StreamingDriftDetector:
    """Page-Hinkley test for distribution drift on streaming losses."""

    def __init__(self, window=500, threshold=50.0, min_samples=100):
        self.threshold, self.min_samples = threshold, min_samples
        self.losses = deque(maxlen=window)
        self.cum_sum = self.min_cum = self.count = self.mean = 0.0

    def update(self, loss: float) -> dict:
        self.count += 1
        self.losses.append(loss)
        self.mean += (loss - self.mean) / self.count
        self.cum_sum += loss - self.mean
        self.min_cum = min(self.min_cum, self.cum_sum)
        ph = self.cum_sum - self.min_cum
        return {"drift": self.count >= self.min_samples and ph > self.threshold,
                "ph_stat": ph, "mean": self.mean}

    def reset(self):
        self.cum_sum = self.min_cum = self.count = self.mean = 0.0
        self.losses.clear()
```

## Gotchas

- **EWC lambda is fragile**: Too high prevents learning new tasks; too low causes forgetting. Start at 1000, tune on held-out task sequence
- **Reservoir sampling bias**: Buffer composition shifts as distribution changes. Monitor buffer class balance periodically
- **Validation gating false negatives**: Short val sets produce noisy scores. Use 500+ examples and consider significance tests before rejecting updates
- **LoRA rank for continual learning**: Higher rank (32-64) adds capacity but increases forgetting risk. Use r=8-16 for incremental updates
- **EMA decay rate**: 0.999 for slow drift; 0.99 for rapid shifts. Too low destabilizes the model
- **Drift detection sensitivity**: Page-Hinkley threshold depends on loss scale. Calibrate on a known-stable period
- **Fisher diagonal is approximate**: Misses parameter interactions. Consider block-diagonal or K-FAC for critical apps
- **Model merging is lossy**: Linear interpolation destroys task-specific features on diverged models. Fisher-weighted helps but requires stored Fisher
