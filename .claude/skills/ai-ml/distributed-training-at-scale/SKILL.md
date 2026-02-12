---
name: distributed-training-at-scale
description: Configure and optimize distributed training across multiple GPUs and nodes
---

# Distributed Training at Scale

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

## DeepSpeed ZeRO Stages

### ZeRO Stage Configs

```python
# ZeRO Stage 1: Shard optimizer states only
# Memory reduction: ~4x optimizer memory
zero1_config = {
    "zero_optimization": {
        "stage": 1,
        "allgather_partitions": True,
        "reduce_scatter": True,
        "overlap_comm": True,  # overlap comm with backward pass
    },
    "bf16": {"enabled": True},
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,
}

# ZeRO Stage 2: Shard optimizer + gradients
# Memory reduction: ~8x vs naive DDP
zero2_config = {
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "reduce_scatter": True,
        "overlap_comm": True,
        "contiguous_gradients": True,  # reduce memory fragmentation
    },
    "bf16": {"enabled": True},
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
}

# ZeRO Stage 3: Shard optimizer + gradients + parameters
# Memory reduction: linear with GPU count
zero3_config = {
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "prefetch_bucket_size": 5e7,       # prefetch params during forward
        "param_persistence_threshold": 1e5, # keep small params unsharded
        "reduce_bucket_size": 5e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_max_live_parameters": 1e9,
    },
    "bf16": {"enabled": True},
}

# ZeRO Stage 3 + CPU Offloading (when GPU memory is exhausted)
zero3_offload_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,     # pinned memory for faster transfers
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
        },
    },
    "bf16": {"enabled": True},
}
```

## FSDP2 with Mixed Precision

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision, ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

def setup_fsdp2_model(model, transformer_layer_cls):
    """Configure FSDP2 with mixed precision and auto-wrapping."""
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,   # compute dtype
        reduce_dtype=torch.bfloat16,  # gradient reduction dtype
        buffer_dtype=torch.bfloat16,
    )
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_cls},
    )
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        auto_wrap_policy=wrap_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # needed for torch.compile compatibility
        limit_all_gathers=True,  # prevent OOM from too many concurrent gathers
    )
    return model
```

## Megatron-LM Tensor and Pipeline Parallelism

```python
# Megatron-LM launch configuration for a 70B model
# 8 nodes x 8 GPUs = 64 GPUs total
# TP=8 (within node), PP=8 (across nodes), DP=1
LAUNCH_CMD = """
python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --nnodes 8 \
    pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 8 \
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --micro-batch-size 1 \
    --global-batch-size 1024 \
    --seq-length 4096 \
    --lr 1.5e-4 \
    --min-lr 1.5e-5 \
    --lr-warmup-iters 2000 \
    --bf16 \
    --use-flash-attn \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --sequence-parallel          # reduces TP activation memory
"""
```

## NCCL Tuning Environment Variables

```python
import os

def set_nccl_env(num_nodes=1):
    """Set NCCL env vars for optimal distributed performance."""
    # Core tuning
    os.environ["NCCL_ALGO"] = "Ring,Tree"        # allow both algorithms
    os.environ["NCCL_PROTO"] = "Simple,LL,LL128"  # all protocols
    os.environ["NCCL_BUFFSIZE"] = str(8 * 1024 * 1024)  # 8MB buffer

    # Multi-node specific
    if num_nodes > 1:
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # network interface
        os.environ["NCCL_IB_DISABLE"] = "0"         # enable InfiniBand
        os.environ["NCCL_NET_GDR_LEVEL"] = "5"      # GPU Direct RDMA level
        os.environ["NCCL_P2P_LEVEL"] = "NVL"        # NVLink P2P

    # Debugging (remove in production)
    os.environ["NCCL_DEBUG"] = "WARN"                # INFO for verbose
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"    # better error messages
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"     # timeout on hangs
```

## Gradient Accumulation with no_sync

```python
from contextlib import nullcontext

def train_step_distributed(model, dataloader, optimizer, scheduler,
                           accumulation_steps=8, max_grad_norm=1.0):
    """Distributed training loop with proper no_sync and accumulation."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        is_accumulating = (step + 1) % accumulation_steps != 0

        # Skip gradient sync during accumulation (saves all-reduce overhead)
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

### Communication Overlap Pattern

```python
# Enable communication/computation overlap in DeepSpeed
ds_config = {
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": True,              # overlap all-reduce with backward
        "reduce_bucket_size": 5e8,         # larger buckets = fewer comms
        "allgather_bucket_size": 5e8,
    },
    "comms_logger": {"enabled": True},     # log comm stats for profiling
}

# In FSDP, overlap is controlled by:
# - forward_prefetch=True: prefetch next FSDP unit's params during forward
# - limit_all_gathers=True: prevents OOM from too many concurrent gathers
# - backward_prefetch=BackwardPrefetch.BACKWARD_PRE: prefetch during backward
```

## Gotchas

- **ZeRO-3 + gradient accumulation**: Must wrap accumulation steps with `model.no_sync()` or DeepSpeed handles it internally -- mixing manual and DS accumulation double-counts
- **FSDP + torch.compile**: Requires `use_orig_params=True`; without it, compile silently falls back to eager
- **TP across nodes**: Tensor parallelism across nodes (non-NVLink) kills throughput -- keep TP intra-node only
- **NCCL timeouts**: Default 30min timeout masks real errors; set `TORCH_NCCL_BLOCKING_WAIT=1` and lower timeout to 5-10min for faster debugging
- **Batch size scaling**: Effective batch = micro_batch * accumulation_steps * dp_world_size. Changing GPU count changes effective batch -- adjust LR accordingly (linear scaling rule)
- **Mixed precision with ZeRO-3**: `bf16` is strongly preferred; `fp16` with ZeRO-3 offload can cause divergence due to master weight precision
- **Checkpoint compatibility**: ZeRO-3 sharded checkpoints require the same world size to reload; convert to consolidated format for portability
- **Pipeline parallelism bubble**: PP introduces idle time (bubble). Minimize with micro-batch count >> PP stages (rule: 4x PP degree minimum)
