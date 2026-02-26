# Generative Model Architectures — Code Reference

## Diffusion Training Loop (DDPM, Epsilon Prediction)

```python
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="scaled_linear",    # "linear", "scaled_linear", "squaredcos_cap_v2"
    beta_start=0.00085,
    beta_end=0.012,
    prediction_type="epsilon",        # "epsilon", "v_prediction", "sample"
)

def training_step(model, vae, text_encoder, batch, weight_dtype=torch.bfloat16):
    with torch.no_grad():
        latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (latents.shape[0],), device=latents.device).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    noise_pred = model(noisy_latents, timesteps, encoder_hidden_states).sample
    loss = F.mse_loss(noise_pred.float(), noise.float())
    return loss
```

## Sampling with Different Schedulers

```python
from diffusers import (
    DDIMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)

scheduler_configs = {
    "ddim_50":    (DDIMScheduler, {"num_inference_steps": 50}),
    "euler_25":   (EulerDiscreteScheduler, {"num_inference_steps": 25}),
    "dpm++_20":   (DPMSolverMultistepScheduler, {"num_inference_steps": 20}),
    "euler_a_30": (EulerAncestralDiscreteScheduler, {"num_inference_steps": 30}),
}

@torch.no_grad()
def sample(pipe, prompt, scheduler_cls, scheduler_kwargs, guidance_scale=7.5):
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    return pipe(
        prompt, guidance_scale=guidance_scale,
        **scheduler_kwargs, generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
```

## Classifier-Free Guidance (CFG) Sampling

```python
@torch.no_grad()
def cfg_sample_step(model, latents, timestep, encoder_hidden_states,
                    guidance_scale=7.5):
    latent_input = torch.cat([latents] * 2)
    timestep_input = torch.cat([timestep] * 2)
    uncond_embeddings = torch.zeros_like(encoder_hidden_states)
    text_input = torch.cat([uncond_embeddings, encoder_hidden_states])

    noise_pred = model(latent_input, timestep_input, text_input).sample
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    guided = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    return guided
```

## ControlNet Conditioning

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

# Training a custom ControlNet (UNet frozen, only ControlNet trains)
controlnet = ControlNetModel.from_unet(pretrained_unet)

def controlnet_training_step(controlnet, unet, batch):
    noisy_latents = add_noise(batch["latents"], noise, timesteps)
    controlnet_cond = batch["conditioning_image"]  # edge map, depth, pose, etc.

    down_samples, mid_sample = controlnet(
        noisy_latents, timesteps, encoder_hidden_states,
        controlnet_cond=controlnet_cond, return_dict=False,
    )
    noise_pred = unet(
        noisy_latents, timesteps, encoder_hidden_states,
        down_block_additional_residuals=down_samples,
        mid_block_additional_residual=mid_sample,
    ).sample
    return F.mse_loss(noise_pred.float(), noise.float())
```

## LoRA for Diffusion Models

```python
from diffusers import StableDiffusionPipeline
from peft import LoraConfig

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
)

unet_lora_config = LoraConfig(
    r=8, lora_alpha=16, init_lora_weights="gaussian",
    target_modules=["to_q", "to_v", "to_k", "to_out.0"],
)
pipe.unet.add_adapter(unet_lora_config)

text_lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
)
pipe.text_encoder.add_adapter(text_lora_config)
```

## Sliding Window Attention

```python
def build_sliding_window_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    """Causal mask with sliding window -- O(n*w) instead of O(n^2)."""
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    causal = col_idx <= row_idx
    window = (row_idx - col_idx) < window_size
    mask = causal & window
    return mask.float().masked_fill(~mask, float("-inf")).masked_fill(mask, 0.0)
```

## Mixture of Experts (MoE) Layer

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

## Parameter Estimation

```python
def estimate_params(d_model: int, n_layers: int, vocab_size: int, n_experts: int = 1) -> int:
    """Quick parameter count estimate."""
    attn = 4 * d_model * d_model * n_layers
    ffn = 8 * d_model * d_model * n_layers * n_experts
    return attn + ffn + vocab_size * d_model
```
