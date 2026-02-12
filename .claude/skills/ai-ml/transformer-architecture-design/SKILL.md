---
name: transformer-architecture-design
description: Design and implement custom transformer architectures for various scales and tasks
---

# Transformer Architecture Design

Custom transformer implementation patterns for production and research. Covers attention variants, positional encodings, mixture-of-experts, state-space hybrids, and compute-optimal scaling decisions.

## Decision Table

| Task / Scale | Attention | Positional Encoding | Architecture | Notes |
|---|---|---|---|---|
| <1B params, general NLP | MHA | RoPE | Dense transformer | Standard baseline |
| 1-7B, long context (>8k) | GQA | RoPE + dynamic NTK | Dense transformer | GQA cuts KV cache 4-8x |
| 7-70B, inference-bound | MQA | ALiBi | Dense transformer | Fastest decode, slight quality tradeoff |
| 70B+, compute-limited | GQA | RoPE | MoE (top-2/8) | 4x effective params at ~2x compute |
| 128k+ context | Sliding window + global | RoPE | Hybrid dense+sparse attn | Mistral/Gemini pattern |
| Real-time sequential | N/A | Learned | SSM (Mamba) | Linear scaling with sequence length |
| Multimodal (vision+text) | MHA cross-attn | 2D RoPE + 1D RoPE | Hybrid ViT + decoder | Separate encodings per modality |
| Code generation | GQA | RoPE | Dense + MoE routing | Sparse experts for language-specific paths |

## Custom Attention Mechanisms

### Multi-Head Attention (MHA)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Standard MHA with flash attention dispatch."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, T, self.d_model))
```

### Grouped Query Attention (GQA)

```python
class GroupedQueryAttention(nn.Module):
    """GQA: n_kv_heads < n_heads, KV heads repeated to match Q heads."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x[:, :, None, :, :].expand(B, H, self.n_rep, T, D).reshape(B, H * self.n_rep, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, -1))
```

### Sliding Window Attention

```python
def build_sliding_window_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    """Causal mask with sliding window — O(n*w) instead of O(n^2)."""
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    causal = col_idx <= row_idx
    window = (row_idx - col_idx) < window_size
    mask = causal & window
    return mask.float().masked_fill(~mask, float("-inf")).masked_fill(mask, 0.0)
```

## Positional Encodings

### Rotary Position Embeddings (RoPE)

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings. x shape: (B, H, T, D)."""
    d_half = x.shape[-1] // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
```

### ALiBi (Attention with Linear Biases)

```python
def build_alibi_bias(n_heads: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """No learned params — add linear distance bias to attention scores."""
    ratio = 2 ** (-8.0 / n_heads)
    slopes = torch.tensor([ratio ** i for i in range(1, n_heads + 1)], device=device)
    positions = torch.arange(seq_len, device=device)
    distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).clamp(min=0).float()
    return -slopes.view(-1, 1, 1) * distances.unsqueeze(0)  # (H, T, T)
```

## Architecture Variations

### Mixture of Experts (MoE) Layer

```python
class MoELayer(nn.Module):
    """Top-k sparse MoE replacing dense FFN."""

    def __init__(self, d_model: int, d_ff: int, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.SiLU(),
                          nn.Linear(d_ff, d_model, bias=False))
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        flat = x.view(-1, D)
        logits = self.gate(flat)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        # Auxiliary load-balance loss
        counts = torch.zeros(self.n_experts, device=x.device)
        counts.scatter_add_(0, indices.view(-1), torch.ones_like(indices.view(-1), dtype=torch.float))
        aux_loss = self.n_experts * ((counts / counts.sum()) * F.softmax(logits, dim=-1).mean(0)).sum()
        out = torch.zeros_like(flat)
        for i in range(self.top_k):
            for e in range(self.n_experts):
                mask = indices[:, i] == e
                if mask.any():
                    out[mask] += weights[mask, i].unsqueeze(-1) * self.experts[e](flat[mask])
        return out.view(B, T, D), aux_loss
```

## Scaling Laws

```python
def chinchilla_optimal(compute_flops: float) -> dict:
    """Compute-optimal N (params) and D (tokens). C ≈ 6*N*D."""
    n_opt = int(0.6 * math.sqrt(compute_flops / 6))
    d_opt = int(compute_flops / (6 * n_opt))
    return {"params": n_opt, "tokens": d_opt, "tokens_per_param": round(d_opt / n_opt, 1)}
# 1e21 FLOPs ≈ 400M params, 8B tokens
# 1e24 FLOPs ≈ 13B params, 260B tokens

def estimate_params(d_model: int, n_layers: int, vocab_size: int, n_experts: int = 1) -> int:
    """Quick parameter count estimate."""
    attn = 4 * d_model * d_model * n_layers
    ffn = 8 * d_model * d_model * n_layers * n_experts
    return attn + ffn + vocab_size * d_model
```

## Gotchas

- **KV cache OOM**: GQA/MQA reduce KV cache linearly with `n_kv_heads`; MHA at 128k context on 7B needs ~16GB just for KV in fp16
- **RoPE extrapolation fails**: models trained at 4k degrade past ~6k without NTK-aware scaling or YaRN; test at 2x training length
- **ALiBi hurts on short seqs**: linear bias penalizes distant tokens regardless; worse than RoPE when context < 2k
- **MoE load imbalance**: without aux loss, >80% tokens route to 1-2 experts; use `aux_loss_weight=0.01` minimum
- **MoE memory**: total params = N_experts * FFN_size — top-2/8 MoE with 7B active holds ~40B total in memory
- **Flash attn masking**: custom masks need explicit `attn_mask`; `is_causal=True` ignores custom masks
- **SSM limitations**: pure Mamba struggles with in-context retrieval; hybrid Mamba+attention mitigates this
- **fp16 overflow**: attention logits overflow at seq > 4k in fp16; use bf16 or fp32 for attention compute
- **Chinchilla is a lower bound**: for inference-optimized models, overtrain 2-5x tokens (LLaMA approach)
