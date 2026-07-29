# PyTorch Distributed Training

## Parallelism Strategy Decision Table

| Model Size | GPUs Available | Strategy | Framework | Key Config |
|-----------|---------------|----------|-----------|------------|
| < 1B | 1-8 | DDP | PyTorch DDP | Straightforward replication |
| 1B-10B | 4-16 | ZeRO-2 + FSDP | DeepSpeed / FSDP2 | Shard grads + optimizer states |
| 10B-70B | 8-64 | ZeRO-3 + TP | DeepSpeed + Megatron | Shard everything, tensor parallel |
| 70B-200B | 32-128 | TP + PP + ZeRO-3 | Megatron-LM | 3D parallelism |
| 200B-1T+ | 128-1024+ | TP + PP + EP + ZeRO-3 | Megatron + DeepSpeed | Full 3D + expert parallelism |

### Choosing Tensor vs Pipeline Parallelism
- **TP**: Split layers across GPUs. Best within a node (high-bandwidth NVLink).
- **PP**: Split layer groups across nodes. Better for cross-node (lower bandwidth).
- Rule of thumb: TP degree = GPUs per node, PP degree = number of nodes.

## Optimizer Configuration

### AdamW Weight Decay Groups
Always separate decay vs no-decay params. Biases, LayerNorm, and embeddings should not be decayed.

```python
def configure_optimizer(model, lr=1e-4, weight_decay=0.01):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "LayerNorm" in name or "layernorm" in name or "embedding" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)
```

### Layer-wise Learning Rate Decay (LLRD)
Use for fine-tuning pretrained models. Lower layers get smaller LR. Typical decay: 0.85-0.95.

## Scheduler Opinions

| Scheduler | When to Use | Steps Per |
|-----------|-------------|-----------|
| `CosineAnnealingLR` | General fine-tuning | **Epoch** |
| `OneCycleLR` | Training from scratch, super-convergence | **Batch** |
| Warmup + Cosine | LLM fine-tuning, large models | **Batch** |

**Gotcha**: OneCycleLR steps per-batch, CosineAnnealingLR steps per-epoch. Mixing these up silently destroys training.

## Gradient Accumulation with no_sync

```python
from contextlib import nullcontext

def train_step_distributed(model, dataloader, optimizer, scheduler,
                           accumulation_steps=8, max_grad_norm=1.0):
    """Effective batch = micro_batch * accumulation_steps * dp_world_size."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        is_accumulating = (step + 1) % accumulation_steps != 0
        sync_context = model.no_sync() if is_accumulating else nullcontext()

        with sync_context:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / accumulation_steps
            loss.backward()

        if not is_accumulating:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
```

## Flash Attention

PyTorch 2.0+ uses Flash Attention via `F.scaled_dot_product_attention`.

```python
output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
```

- Shape: `(batch, heads, seq_len, head_dim)`, `head_dim <= 256`, SM80+
- `is_causal=True` faster than explicit mask; dropout only in training mode
- `torch.backends.cuda.sdp_kernel()` to force specific backend for debugging

## DDP Setup with torchrun

```python
def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = DDP(create_model().cuda(), device_ids=[local_rank])
    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, num_workers=4, pin_memory=True,
    )

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # CRITICAL: without this, data order repeats
        train_one_epoch(model, train_loader, optimizer)
        if dist.get_rank() == 0:
            torch.save(model.module.state_dict(), f"ckpt_{epoch}.pt")
    dist.destroy_process_group()
```

## FSDP Sharding Strategies

| Strategy | Memory Savings | Communication | Use When |
|----------|---------------|---------------|----------|
| `FULL_SHARD` | Maximum | Highest | Model barely fits across GPUs |
| `SHARD_GRAD_OP` | Moderate | Lower | Model fits but optimizer doesn't |
| `NO_SHARD` | None (like DDP) | Lowest | Debugging FSDP issues |
| `HYBRID_SHARD` | Per-node full, cross-node shard | Balanced | Multi-node with fast intra-node |

Use `SHARDED_STATE_DICT` during training. `FULL_STATE_DICT` only for final export to single-GPU inference.

> See `ai-ml/pytorch-distributed-training/fsdp-deepspeed-megatron.md` for full FSDP2, DeepSpeed ZeRO stage configs, Megatron-LM launch commands, and NCCL tuning.

## Key Pitfalls Checklist

### Training Correctness
- [ ] `model.train()` before training, `model.eval()` before eval
- [ ] `optimizer.zero_grad(set_to_none=True)` -- more memory-efficient
- [ ] Loss divided by `accumulation_steps` when using gradient accumulation
- [ ] `torch.no_grad()` AND `model.eval()` during validation (both required)

### Memory Leaks
- [ ] `.detach()` tensors before appending to lists
- [ ] `.item()` for scalar logging
- [ ] Delete intermediate tensors and `torch.cuda.empty_cache()` if OOM during eval

### Distributed Training
- [ ] `sampler.set_epoch(epoch)` every epoch in DDP
- [ ] `model.no_sync()` during gradient accumulation steps
- [ ] Save checkpoints on `rank == 0` only
- [ ] `model.module` to access underlying model in DDP
- [ ] `find_unused_parameters=True` if model has conditional branches
- [ ] `NCCL_ASYNC_ERROR_HANDLING=1` for better error messages

### Numeric Stability
- [ ] Gradient clipping before optimizer step (`clip_grad_norm_` max: 1.0)
- [ ] `bfloat16` over `float16` when hardware supports it
- [ ] Watch for NaN in loss -- often LR too high or data issues
- [ ] `torch.compile` can change numerics slightly -- validate against eager mode

### Performance
- [ ] `pin_memory=True` in DataLoader for GPU training
- [ ] `non_blocking=True` on `.to(device)` calls with pinned memory
- [ ] `num_workers > 0` (typically 4-8 per GPU), `persistent_workers=True`
- [ ] `torch.compile(model)` for 10-30% speedup on PyTorch 2.x
- [ ] `torch.set_float32_matmul_precision('medium')` on Ampere+ for TF32

## Gotchas

- **ZeRO-3 + gradient accumulation**: Must use `model.no_sync()` or let DeepSpeed handle it internally -- mixing manual and DS accumulation double-counts
- **FSDP + torch.compile**: Requires `use_orig_params=True`; without it, compile silently falls back to eager
- **TP across nodes**: Tensor parallelism across nodes (non-NVLink) kills throughput -- keep TP intra-node only
- **NCCL timeouts**: Default 30min masks errors; set `TORCH_NCCL_BLOCKING_WAIT=1` and lower to 5-10min
- **Batch size scaling**: Effective batch = micro_batch * accumulation_steps * dp_world_size; changing GPU count changes effective batch -- adjust LR (linear scaling rule)
- **Mixed precision with ZeRO-3**: `bf16` strongly preferred; `fp16` with ZeRO-3 offload can diverge
- **Checkpoint compatibility**: ZeRO-3 sharded checkpoints require same world size to reload; convert to consolidated for portability
- **Pipeline parallelism bubble**: Minimize with micro-batch count >> PP stages (4x PP degree minimum)
- **DDP unused params**: Unused parameters cause hangs -- `find_unused_parameters=True` or fix the model
- **FSDP auto_wrap_policy**: Must match model's layer class -- wrong wrapping = OOM or no sharding
- **FSDP cpu_offload**: Saves GPU memory but 5-10x slower; last resort only
