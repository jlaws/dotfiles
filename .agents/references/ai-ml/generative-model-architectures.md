# Generative Model Architectures

## Architecture Decision Table

| Goal | Architecture | Attention | Pos Encoding | Notes |
|------|-------------|-----------|-------------|-------|
| Image gen (SD ecosystem compat) | UNet + DDPM/DDIM | Conv-based | N/A | ControlNet compatible |
| Image gen (new project) | DiT + Flow Matching | MHA/GQA | 2D RoPE | State-of-art; SD3, Flux |
| Real-time image gen | DiT + Rectified Flow | GQA | 2D RoPE | 1-4 step distilled |
| LLM <1B params | Dense transformer | MHA | RoPE | Standard baseline |
| LLM 1-7B, long ctx (>8k) | Dense transformer | GQA | RoPE + dynamic NTK | GQA cuts KV cache 4-8x |
| LLM 7-70B, inference-bound | Dense transformer | MQA | ALiBi | Fastest decode |
| LLM 70B+ | MoE (top-2/8) | GQA | RoPE | 4x params at ~2x compute |
| 128k+ context | Hybrid dense+sparse | Sliding window + global | RoPE | Mistral/Gemini pattern |
| Real-time sequential | SSM (Mamba) | N/A | Learned | Linear scaling w/ seq len |
| Multimodal (vision+text) | ViT + decoder | MHA cross-attn | 2D + 1D RoPE | Separate encodings per modality |

## Diffusion Formulations

| Formulation | Training | Sampling | When to Use |
|-------------|----------|----------|-------------|
| **DDPM** | Discrete timesteps, epsilon pred | Slow (1000 steps) | Learning/prototyping |
| **DDIM** | Same as DDPM | Fast (10-50 steps, deterministic) | Drop-in faster DDPM sampling |
| **Flow Matching** | Continuous time, velocity pred | Fast, ODE-based | State-of-art; SD3, Flux |
| **Rectified Flow** | Straight paths, reflow | Very fast (1-4 steps) | Distilled, real-time inference |

### Scheduler Selection (Inference)

| Scheduler | Steps | Quality | Speed | Notes |
|-----------|-------|---------|-------|-------|
| DDIM | 50 | Good | Slow | Deterministic, invertible |
| Euler | 20-30 | Good | Fast | Reliable default |
| DPM++ 2M Karras | 20 | Great | Fast | Best quality/speed tradeoff |
| Euler Ancestral | 25-30 | Good + varied | Fast | Stochastic, more diverse |
| LCM | 4-8 | Decent | Very fast | Requires LCM-LoRA or distilled model |

### CFG Scale Guidelines

| Scale | Effect | Use Case |
|-------|--------|----------|
| 1.0 | No guidance | Diversity exploration |
| 3.0-5.0 | Mild | Artistic, less saturated |
| 7.0-8.5 | Standard | General purpose |
| 10.0-15.0 | Strong | Precise prompt following |
| 15.0+ | Over-saturated | Usually artifacts |

## Transformer Attention Variants

### Multi-Head Attention (MHA)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
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
        return self.out_proj(out.transpose(1, 2).reshape(B, T, -1))
```

### Grouped Query Attention (GQA)

```python
class GroupedQueryAttention(nn.Module):
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

## Positional Encodings

### RoPE

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
    d_half = x.shape[-1] // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
```

### ALiBi

```python
def build_alibi_bias(n_heads: int, seq_len: int, device: torch.device) -> torch.Tensor:
    ratio = 2 ** (-8.0 / n_heads)
    slopes = torch.tensor([ratio ** i for i in range(1, n_heads + 1)], device=device)
    positions = torch.arange(seq_len, device=device)
    distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).clamp(min=0).float()
    return -slopes.view(-1, 1, 1) * distances.unsqueeze(0)
```

## Scaling Laws

```python
import math

def chinchilla_optimal(compute_flops: float) -> dict:
    """Compute-optimal N (params) and D (tokens). C ~ 6*N*D with D ~ 20*N,
    so C = 120*N**2 and N = sqrt(C/120) -- about 20 tokens per parameter."""
    n_opt = int(math.sqrt(compute_flops / 120))
    d_opt = int(compute_flops / (6 * n_opt))
    return {"params": n_opt, "tokens": d_opt, "tokens_per_param": round(d_opt / n_opt, 1)}
```

## Gotchas

### Diffusion
- `scaled_linear` beta schedule is SD1/SD2 default; `squaredcos_cap_v2` better for high-res
- v-prediction requires matching schedule at training AND inference -- cannot swap to epsilon
- Always multiply latents by `vae.config.scaling_factor` (0.18215 SD1.x, 0.13025 SDXL)
- Use EMA decay 0.9999 for >100M params; start after warmup; evaluate EMA model not training model
- LR above 1e-4 for diffusion fine-tuning is too high -- 1e-5 to 5e-5 typical
- Freeze VAE and text encoder during UNet training
- `pipe.enable_model_cpu_offload()` + `torch.compile(pipe.unet)` for inference optimization

### Transformer
- **KV cache OOM**: GQA/MQA reduce KV cache linearly with `n_kv_heads`
- **RoPE extrapolation fails** past ~1.5x training length without NTK scaling or YaRN
- **ALiBi hurts on short seqs** (<2k) -- linear bias penalizes distant tokens regardless
- **MoE load imbalance**: without aux loss, >80% tokens route to 1-2 experts; use `aux_loss_weight=0.01`
- **MoE memory**: total params = N_experts * FFN_size -- top-2/8 with 7B active holds ~40B total
- **fp16 overflow**: attention logits overflow at seq >4k; use bf16 or fp32 for attention compute
- **Chinchilla is a lower bound**: for inference-optimized models, overtrain 2-5x tokens (LLaMA approach)

## Neural Architecture Search

Automate model design with differentiable search (DARTS), RL-based controllers (ENAS), one-shot supernets, and hardware-aware latency optimization (ProxylessNAS). Covers search space design, alternating optimization, weight sharing, and latency lookup tables.

For NAS patterns, see `ai-ml/generative-model-architectures/neural-architecture-search.md`.

## Extended References

See `references/` for:
- Code examples: diffusion training loops, CFG sampling, ControlNet, LoRA, sliding window attention, MoE layer (`code-examples.md`)
- Neural architecture search: DARTS, ENAS, SuperNet, ProxylessNAS (`neural-architecture-search.md`)
