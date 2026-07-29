# FSDP, DeepSpeed ZeRO, and Megatron-LM Reference

## DeepSpeed ZeRO Stage Configs

```python
# ZeRO Stage 1: Shard optimizer states only
# Memory reduction: ~4x optimizer memory
zero1_config = {
    "zero_optimization": {
        "stage": 1,
        "allgather_partitions": True,
        "reduce_scatter": True,
        "overlap_comm": True,
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
        "contiguous_gradients": True,
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
        "prefetch_bucket_size": 5e7,
        "param_persistence_threshold": 1e5,
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
            "pin_memory": True,
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
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
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
        limit_all_gathers=True,  # prevent OOM from concurrent gathers
    )
    return model
```

## FSDP Checkpointing

```python
from torch.distributed.fsdp import StateDictType

def save_fsdp_checkpoint(model, optimizer, path):
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state = {"model": model.state_dict(),
                 "optim": FSDP.optim_state_dict(model, optimizer)}
        torch.distributed.checkpoint.save_state_dict(state, checkpoint_id=path)
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
    --sequence-parallel
"""
```

## NCCL Tuning Environment Variables

```python
import os

def set_nccl_env(num_nodes=1):
    """Set NCCL env vars for optimal distributed performance."""
    os.environ["NCCL_ALGO"] = "Ring,Tree"
    os.environ["NCCL_PROTO"] = "Simple,LL,LL128"
    os.environ["NCCL_BUFFSIZE"] = str(8 * 1024 * 1024)

    if num_nodes > 1:
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
        os.environ["NCCL_IB_DISABLE"] = "0"
        os.environ["NCCL_NET_GDR_LEVEL"] = "5"
        os.environ["NCCL_P2P_LEVEL"] = "NVL"

    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
```

## Communication Overlap Pattern

```python
# DeepSpeed communication/computation overlap
ds_config = {
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": True,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
    },
    "comms_logger": {"enabled": True},
}

# In FSDP, overlap is controlled by:
# - forward_prefetch=True: prefetch next FSDP unit's params during forward
# - limit_all_gathers=True: prevents OOM from too many concurrent gathers
# - backward_prefetch=BackwardPrefetch.BACKWARD_PRE: prefetch during backward
```

## Layer-wise Learning Rate Decay (LLRD)

Scale each layer's LR by `layer_decay ** (depth_from_top)` so early layers move least.
Group by **layer depth**, not by position in `named_parameters()` -- the two are not the
same, and indexing off the parameter enumeration produces negative exponents (and so LRs
*above* `base_lr`) for every parameter past the module count. Exclude bias and norm
parameters from weight decay.

## Warmup + Cosine Scheduler

Linear warmup for `warmup_steps`, then cosine decay to `min_lr_ratio * base_lr` over the
remaining steps, via `torch.optim.lr_scheduler.LambdaLR`.
