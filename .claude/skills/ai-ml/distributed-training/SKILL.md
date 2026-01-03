---
name: distributed-training
description: Master distributed training patterns including multi-GPU, DDP, FSDP, and gradient accumulation for large-scale ML training. Use when training large models, scaling to multiple GPUs, or optimizing memory usage.
---

# Distributed Training

Master distributed training patterns for scaling ML experiments from single GPU to multi-node clusters.

## When to Use This Skill

- Training models that don't fit on a single GPU
- Scaling training to multiple GPUs for speed
- Implementing data parallel or model parallel training
- Optimizing memory usage for large models
- Setting up multi-node distributed training

## Single GPU Optimization

### Memory Optimization Techniques
```python
import torch
from torch.cuda.amp import autocast, GradScaler

def memory_efficient_training(model, dataloader, optimizer):
    """Single GPU training with memory optimizations."""

    scaler = GradScaler()  # Mixed precision

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)  # More memory efficient

        # Mixed precision forward pass
        with autocast():
            outputs = model(batch["input"].cuda())
            loss = criterion(outputs, batch["target"].cuda())

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Clear cache periodically (use sparingly)
        # torch.cuda.empty_cache()


# Enable memory-efficient attention (PyTorch 2.0+)
import torch.nn.functional as F

def efficient_attention(q, k, v):
    """Use Flash Attention when available."""
    return F.scaled_dot_product_attention(q, k, v)
```

### Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class MemoryEfficientTransformer(torch.nn.Module):
    def __init__(self, layers, use_checkpointing=True):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.use_checkpointing = use_checkpointing

    def forward(self, x):
        if self.use_checkpointing and self.training:
            # Checkpoint each layer
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x)
        return x


# For sequential models
def forward_with_checkpoint(model, x, chunks=4):
    """Checkpoint sequential model in chunks."""
    return checkpoint_sequential(model, chunks, x, use_reentrant=False)
```

## Data Parallel (DP) - Simple Multi-GPU

### Basic DataParallel
```python
import torch.nn as nn

# Simple but not recommended for serious training
model = nn.DataParallel(model)
model = model.cuda()

# All GPUs process different batches, gradients averaged on GPU 0
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
```

**Limitations of DataParallel:**
- Imbalanced GPU memory (GPU 0 does more work)
- GIL bottleneck in Python
- Communication overhead

## Distributed Data Parallel (DDP) - Recommended

### DDP Setup
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

def setup_ddp(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP resources."""
    dist.destroy_process_group()


def train_ddp(rank: int, world_size: int, config: dict):
    """Training function for each DDP process."""
    setup_ddp(rank, world_size)

    # Create model and wrap with DDP
    model = create_model(config).cuda(rank)
    model = DDP(model, device_ids=[rank])

    # Use DistributedSampler for data
    train_dataset = create_dataset(config)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        sampler.set_epoch(epoch)  # Important for shuffling

        for batch in train_loader:
            inputs = batch["input"].cuda(rank)
            targets = batch["target"].cuda(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Only save on rank 0
        if rank == 0:
            torch.save(model.module.state_dict(), f"checkpoint_epoch_{epoch}.pt")

    cleanup_ddp()


# Launch training
import torch.multiprocessing as mp

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size)
```

### DDP with torchrun (Recommended)
```python
# train_ddp.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # torchrun sets these automatically
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    # Create model
    model = create_model().cuda()
    model = DDP(model, device_ids=[local_rank])

    # Training loop...

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

```bash
# Launch with torchrun (single node, 4 GPUs)
torchrun --nproc_per_node=4 train_ddp.py

# Multi-node (2 nodes, 4 GPUs each)
# Node 0:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
         --master_addr=node0 --master_port=12355 train_ddp.py

# Node 1:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
         --master_addr=node0 --master_port=12355 train_ddp.py
```

## Fully Sharded Data Parallel (FSDP) - Large Models

### FSDP Setup
```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import functools

def create_fsdp_model(model, config):
    """Wrap model with FSDP for large model training."""

    # Mixed precision policy
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Auto-wrap policy for transformers
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            torch.nn.TransformerEncoderLayer,
            # Add your transformer block class
        },
    )

    # Or size-based wrapping
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=100_000_000,  # 100M params
    )

    # Wrap with FSDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=False),  # Enable for very large models
        device_id=torch.cuda.current_device(),
    )

    return model


def train_fsdp():
    """Training loop with FSDP."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = create_model()
    model = create_fsdp_model(model, config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch["input"].cuda())
                loss = criterion(outputs, batch["target"].cuda())

            loss.backward()
            optimizer.step()

    # Save full model (only on rank 0)
    if dist.get_rank() == 0:
        # Need to gather all shards
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state_dict = model.state_dict()
            torch.save(state_dict, "model.pt")

    dist.destroy_process_group()
```

### FSDP Checkpointing
```python
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.checkpoint import save_state_dict, load_state_dict

def save_fsdp_checkpoint(model, optimizer, path):
    """Save FSDP checkpoint efficiently."""

    # Sharded state dict (saves memory)
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    ):
        state_dict = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
        }

        save_state_dict(state_dict, checkpoint_id=path)


def load_fsdp_checkpoint(model, optimizer, path):
    """Load FSDP checkpoint."""

    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    ):
        state_dict = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
        }

        load_state_dict(state_dict, checkpoint_id=path)

        model.load_state_dict(state_dict["model"])
        FSDP.optim_state_dict_to_load(model, optimizer, state_dict["optimizer"])
```

## Gradient Accumulation

### Effective Batch Size Scaling
```python
def train_with_accumulation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accumulation_steps: int = 4,
):
    """
    Gradient accumulation for larger effective batch size.

    Effective batch = batch_size * accumulation_steps * world_size
    """
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # Forward pass
        outputs = model(batch["input"].cuda())
        loss = criterion(outputs, batch["target"].cuda())

        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()

        # Update weights every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            # Optional: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

    # Handle remaining gradients
    if (step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
```

### DDP with Gradient Accumulation
```python
from contextlib import nullcontext

def train_ddp_with_accumulation(model, dataloader, optimizer, accumulation_steps):
    """DDP training with gradient accumulation."""

    for step, batch in enumerate(dataloader):
        # Only sync gradients on accumulation boundary
        is_accumulating = (step + 1) % accumulation_steps != 0

        # Skip gradient sync during accumulation
        context = model.no_sync() if is_accumulating else nullcontext()

        with context:
            outputs = model(batch["input"].cuda())
            loss = criterion(outputs, batch["target"].cuda())
            loss = loss / accumulation_steps
            loss.backward()

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
```

## PyTorch Lightning Integration

### Automatic Distributed Training
```python
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = create_model(config)
        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["input"])
        loss = criterion(outputs, batch["target"])
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


# DDP training
trainer = pl.Trainer(
    accelerator="gpu",
    devices=4,  # 4 GPUs
    strategy="ddp",
    precision="16-mixed",
    max_epochs=100,
)

# FSDP training for large models
trainer = pl.Trainer(
    accelerator="gpu",
    devices=4,
    strategy=FSDPStrategy(
        sharding_strategy="FULL_SHARD",
        mixed_precision="bf16",
    ),
    precision="bf16-mixed",
)

# Multi-node
trainer = pl.Trainer(
    accelerator="gpu",
    devices=4,
    num_nodes=2,
    strategy="ddp",
)

trainer.fit(model, train_loader)
```

## Communication Primitives

### Collective Operations
```python
import torch.distributed as dist

def distributed_metrics(tensor, world_size):
    """Aggregate metrics across processes."""

    # All-reduce: sum across all processes
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / world_size  # Average

    return tensor


def gather_results(tensor, world_size, rank):
    """Gather results from all processes to rank 0."""

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.gather(tensor, gathered if rank == 0 else None, dst=0)

    if rank == 0:
        return torch.cat(gathered)
    return None


def broadcast_config(config, rank):
    """Broadcast config from rank 0 to all."""

    if rank == 0:
        config_tensor = torch.tensor([config["lr"], config["batch_size"]])
    else:
        config_tensor = torch.zeros(2)

    dist.broadcast(config_tensor, src=0)

    return {
        "lr": config_tensor[0].item(),
        "batch_size": int(config_tensor[1].item()),
    }
```

## Best Practices

1. **Start with DDP** before moving to FSDP
2. **Use torchrun** for launching distributed jobs
3. **Set epochs on sampler** for proper shuffling
4. **Only save on rank 0** to avoid file conflicts
5. **Use sync_dist=True** when logging in distributed
6. **Profile communication** to identify bottlenecks

## Common Pitfalls

- **Forgetting sampler.set_epoch()**: Breaks shuffling in DDP
- **Not handling rank 0 only operations**: Deadlocks or duplicate work
- **Wrong batch size calculation**: Effective batch = local * world_size
- **Model state dict confusion**: DDP wraps with `.module`, FSDP shards
- **NCCL timeout**: Increase timeout for large models or slow networks
