# Neural Architecture Search

## Decision Table

| Approach | Compute Budget | Target Hardware | Best For |
|----------|---------------|-----------------|----------|
| **DARTS** | Low (1-4 GPU-days) | Any | CNN cells, quick iteration |
| **ENAS** | Low (0.5 GPU-days) | Any | RNN/CNN with weight sharing |
| **One-Shot (SuperNet)** | Medium (4-10 GPU-days) | Any | Large search spaces |
| **ProxylessNAS** | Medium (4-8 GPU-days) | Mobile/Edge | Latency-constrained deploy |
| **Random Search** | Any | Any | Baseline, surprisingly strong |
| **Hardware-Aware NAS** | Medium-High | Specific target | Production deployment |

## Search Space Design

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard NAS operation set (DARTS-style)
OPS = {
    "none": lambda C, stride: Zero(stride),
    "skip_connect": lambda C, stride: (
        nn.Identity() if stride == 1 else FactorizedReduce(C, C)
    ),
    "sep_conv_3x3": lambda C, stride: SepConv(C, C, 3, stride, 1),
    "sep_conv_5x5": lambda C, stride: SepConv(C, C, 5, stride, 2),
    "dil_conv_3x3": lambda C, stride: DilConv(C, C, 3, stride, 2, 2),
    "avg_pool_3x3": lambda C, stride: nn.AvgPool2d(3, stride, 1),
    "max_pool_3x3": lambda C, stride: nn.MaxPool2d(3, stride, 1),
}

class SepConv(nn.Module):
    """Separable convolution: depthwise + pointwise with BN-ReLU."""
    def __init__(self, C_in, C_out, kernel, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel, stride, padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        return self.op(x)
```

## DARTS: Differentiable Architecture Search

### Mixed Operations with Architecture Parameters

Relax the discrete choice into a softmax-weighted sum over all candidate ops per edge, with
one `alphas` parameter matrix of shape `(num_edges, len(OPS))` per cell type (normal and
reduction), initialized near zero. `num_edges` for `n` intermediate nodes is
`sum(i + 2 for i in range(n))` -- the `+2` is the two cell input nodes.

Train by **alternating optimization**: architecture parameters on a held-out validation
batch, network weights on the training batch. Using the training batch for both lets the
architecture parameters overfit and is the usual cause of a degenerate search. First-order
alternation is the cheap variant; DARTS' second-order variant differentiates through the
weight update and costs substantially more per step.

## ENAS: RL-Based Controller

An LSTM controller samples a connection and an op per node, and is trained with REINFORCE
against held-out accuracy, using a moving-average baseline for variance reduction and
gradient clipping on the controller. Use held-out accuracy, never training accuracy, as the
reward -- see Gotchas.

## One-Shot SuperNet Training

Build one weight-sharing network holding every candidate op, then train a single uniformly
sampled path per step. Sampling uniformly (rather than by the controller's current
distribution) keeps the shared weights from co-adapting to whichever path is currently
favored. SuperNet rankings do not transfer directly to standalone accuracy -- retrain the
top-k from scratch before believing them.

## Hardware-Aware NAS

### Latency Lookup Table

```python
import time

def build_latency_table(ops_dict, input_shape, device, n_runs=100):
    """Profile each op to build latency lookup table."""
    table = {}
    x = torch.randn(1, *input_shape).to(device)
    for name, op_fn in ops_dict.items():
        op = op_fn(input_shape[0], stride=1).to(device).eval()
        for _ in range(10):  # warmup
            op(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            op(x)
        torch.cuda.synchronize()
        table[name] = (time.perf_counter() - start) / n_runs
    return table

def latency_loss(architecture, latency_table, target_ms=5.0, lambda_lat=0.1):
    """Differentiable latency penalty using softmax-weighted lookup."""
    total = sum(
        sum(w * latency_table[op] for w, op in zip(F.softmax(a, dim=-1), OPS))
        for a in architecture
    )
    return lambda_lat * max(0, total - target_ms)
```

### ProxylessNAS Path Binarization

```python
class ProxylessMixedOp(nn.Module):
    """Memory-efficient: only two paths active during training."""
    def __init__(self, C, stride, ops_list):
        super().__init__()
        self.ops = nn.ModuleList(ops_list)
        self.alpha = nn.Parameter(torch.zeros(len(ops_list)))

    def forward(self, x):
        probs = F.softmax(self.alpha, dim=0)
        idx = torch.multinomial(probs, 2, replacement=False)
        w0 = probs[idx[0]] / (probs[idx[0]] + probs[idx[1]])
        w1 = probs[idx[1]] / (probs[idx[0]] + probs[idx[1]])
        return w0 * self.ops[idx[0]](x) + w1 * self.ops[idx[1]](x)

    def derive_architecture(self):
        return torch.argmax(self.alpha).item()
```

## Gotchas

- **DARTS collapse**: Converges to skip connections only; add edge normalization or operation dropout
- **Weight sharing bias**: SuperNet rankings don't match standalone; always retrain top-k from scratch
- **Search space > algorithm**: Random search in a good space beats NAS in a bad space
- **Proxy tasks mislead**: CIFAR-10 results don't transfer to ImageNet without careful space design
- **Latency tables are device-specific**: Rebuild per target device; batch size affects rankings
- **Memory explosion in DARTS**: All ops run simultaneously; use progressive search or channel pruning
- **Discrete-continuous gap**: Softmax relaxation != argmax; validate by training derived architecture
- **Controller reward hacking**: Use held-out accuracy, not training accuracy, as ENAS reward
