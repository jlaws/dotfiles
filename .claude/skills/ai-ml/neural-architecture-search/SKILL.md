---
name: neural-architecture-search
description: Automate model design with NAS algorithms, search spaces, and hardware-aware optimization
---

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

```python
class MixedOp(nn.Module):
    """Weighted sum of candidate operations (continuous relaxation)."""
    def __init__(self, C, stride):
        super().__init__()
        self.ops = nn.ModuleList([OPS[name](C, stride) for name in OPS])

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class DARTSCell(nn.Module):
    """Single cell with learnable topology (normal or reduction)."""
    def __init__(self, C, num_nodes=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.ops = nn.ModuleDict()
        for node in range(num_nodes):
            for prev in range(node + 2):  # +2 for two input nodes
                self.ops[f"{node}_{prev}"] = MixedOp(C, stride=1)

    def forward(self, s0, s1, alphas):
        states = [s0, s1]
        offset = 0
        for node in range(self.num_nodes):
            node_inputs = []
            for prev in range(len(states)):
                weights = F.softmax(alphas[offset + prev], dim=-1)
                node_inputs.append(self.ops[f"{node}_{prev}"](states[prev], weights))
            states.append(sum(node_inputs))
            offset += len(states) - 1
        return torch.cat(states[-self.num_nodes:], dim=1)

class DARTSNetwork(nn.Module):
    """Full supernet with architecture parameters."""
    def __init__(self, C_init=16, num_cells=8, num_classes=10, num_nodes=4):
        super().__init__()
        num_edges = sum(i + 2 for i in range(num_nodes))
        # Architecture params (learned via second-order optimization)
        self.alphas_normal = nn.Parameter(torch.randn(num_edges, len(OPS)) * 1e-3)
        self.alphas_reduce = nn.Parameter(torch.randn(num_edges, len(OPS)) * 1e-3)
        self.cells = nn.ModuleList()  # build cells here
        self.classifier = nn.Linear(C_init * num_nodes, num_classes)

def darts_train_step(network, arch_optimizer, weight_optimizer,
                     train_batch, val_batch):
    """Alternating optimization: weights on train, arch params on val."""
    arch_optimizer.zero_grad()
    val_loss = F.cross_entropy(network(val_batch["x"]), val_batch["y"])
    val_loss.backward()
    arch_optimizer.step()

    weight_optimizer.zero_grad()
    train_loss = F.cross_entropy(network(train_batch["x"]), train_batch["y"])
    train_loss.backward()
    weight_optimizer.step()
    return train_loss.item(), val_loss.item()
```

## ENAS: RL-Based Controller

```python
class ENASController(nn.Module):
    """LSTM controller that samples architectures via RL."""
    def __init__(self, num_nodes=4, num_ops=8, hidden_size=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.op_emb = nn.Embedding(num_ops, hidden_size)
        self.node_op_head = nn.Linear(hidden_size, num_ops)
        self.node_conn_head = nn.Linear(hidden_size, num_nodes)

    def sample(self):
        """Sample one architecture; return (actions, log_probs)."""
        actions, log_probs = [], []
        h, c, inp = torch.zeros(1, 64), torch.zeros(1, 64), torch.zeros(1, 64)
        for node in range(self.num_nodes):
            h, c = self.lstm(inp, (h, c))
            conn_dist = torch.distributions.Categorical(
                logits=self.node_conn_head(h)[:, :node + 2])
            conn = conn_dist.sample()
            h, c = self.lstm(inp, (h, c))
            op_dist = torch.distributions.Categorical(logits=self.node_op_head(h))
            op = op_dist.sample()
            actions.append((conn.item(), op.item()))
            log_probs.append(conn_dist.log_prob(conn) + op_dist.log_prob(op))
            inp = self.op_emb(op)
        return actions, torch.stack(log_probs)

    def reinforce_step(self, reward, log_probs, baseline, optimizer):
        """REINFORCE with baseline for variance reduction."""
        loss = -((reward - baseline) * log_probs.sum())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        optimizer.step()
```

## One-Shot SuperNet Training

```python
class SuperNet(nn.Module):
    """Weight-sharing supernet: all paths share parameters."""
    def __init__(self, C=16, num_nodes=4):
        super().__init__()
        self.nodes = nn.ModuleList()
        for node in range(num_nodes):
            node_ops = nn.ModuleDict()
            for prev in range(node + 2):
                node_ops[str(prev)] = nn.ModuleList(
                    [OPS[name](C, 1) for name in OPS])
            self.nodes.append(node_ops)

    def forward(self, x, architecture):
        """Forward with a specific sampled architecture."""
        states = [x, x]
        for node_ops, (chosen_input, chosen_op) in zip(self.nodes, architecture):
            states.append(node_ops[str(chosen_input)][chosen_op](states[chosen_input]))
        return torch.cat(states[-len(self.nodes):], dim=1)

def train_supernet_epoch(supernet, loader, optimizer, controller):
    """Uniform path sampling: train one random path per step."""
    supernet.train()
    for batch_x, batch_y in loader:
        arch = controller.sample_uniform()  # random architecture
        optimizer.zero_grad()
        loss = F.cross_entropy(supernet(batch_x.cuda(), arch), batch_y.cuda())
        loss.backward()
        optimizer.step()
```

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
