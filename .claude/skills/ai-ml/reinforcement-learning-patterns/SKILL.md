---
name: reinforcement-learning-patterns
description: RLHF, DPO, PPO, and reward modeling patterns for aligning LLMs. Use when implementing preference-based training or choosing between alignment methods.
---

# Reinforcement Learning Patterns for LLMs

## Method Selection

| Method | Data Required | Compute | Stability | When to Use |
|--------|--------------|---------|-----------|-------------|
| **RLHF (PPO)** | Pairwise prefs + reward model | High (4 models in memory) | Fragile | Maximum control over reward shaping |
| **DPO** | Pairwise preferences | Low (2 models) | Stable | Default choice; simple and effective |
| **KTO** | Binary signal (good/bad) | Low (2 models) | Stable | No pairwise data, only thumbs up/down |
| **ORPO** | Pairwise preferences | Lowest (1 model) | Most stable | Don't want reference model overhead |
| **SimPO** | Pairwise preferences | Lowest (1 model) | Stable | Reference-free + length-normalized; often outperforms DPO |
| **CPO** | Pairwise preferences | Low (2 models) | Stable | Want DPO-like with explicit preference margin |
| **IPO** | Pairwise preferences | Low (2 models) | More stable than DPO | DPO overfitting to preferences |

Rule of thumb: start with DPO. Move to PPO only if you need a shaped reward signal that pairwise preferences can't capture.

## Preference Dataset Creation

### Format

```python
# Standard preference format for TRL
preference_example = {
    "prompt": "Explain quantum computing simply.",
    "chosen": "Quantum computers use qubits that can be 0, 1, or both...",
    "rejected": "Quantum computing is a type of computation that harnesses...",
}
```

### Building from Completions

```python
from datasets import Dataset

def build_preference_dataset(prompts, completions_a, completions_b, labels):
    """labels[i] = 'a' if completions_a[i] preferred, else 'b'"""
    records = []
    for prompt, a, b, label in zip(prompts, completions_a, completions_b, labels):
        chosen, rejected = (a, b) if label == "a" else (b, a)
        records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return Dataset.from_list(records)
```

### Chat-Formatted Preferences

```python
# TRL expects chat-formatted messages for chat models
preference_example_chat = {
    "chosen": [
        {"role": "user", "content": "Explain quantum computing."},
        {"role": "assistant", "content": "Qubits can represent 0 and 1 simultaneously..."},
    ],
    "rejected": [
        {"role": "user", "content": "Explain quantum computing."},
        {"role": "assistant", "content": "It's complicated but basically..."},
    ],
}
```

## Reward Model Training

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B", num_labels=1, torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

config = RewardConfig(
    output_dir="reward_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1.5e-5,
    num_train_epochs=1,           # 1 epoch to avoid overfitting
    bf16=True,
    max_length=1024,
    logging_steps=10,
)

trainer = RewardTrainer(
    model=model, tokenizer=tokenizer, config=config,
    train_dataset=preference_dataset,
)
trainer.train()
```

## DPO with TRL

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

dpo_config = DPOConfig(
    output_dir="dpo_output",
    beta=0.1,                      # KL penalty strength
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=1,
    bf16=True,
    max_length=1024,
    max_prompt_length=512,
    logging_steps=10,
    loss_type="sigmoid",           # sigmoid (standard) or hinge
)

trainer = DPOTrainer(
    model=model, ref_model=None,   # None = use implicit ref via peft
    config=dpo_config, tokenizer=tokenizer,
    train_dataset=preference_dataset,
    peft_config=peft_config,
)
trainer.train()
```

### DPO Beta Tuning

| Beta | Effect | Use When |
|------|--------|----------|
| 0.05 | Weak KL constraint, more deviation from base | Strong preference signal, want big changes |
| 0.1 | Standard | Default starting point |
| 0.3 | Strong KL constraint, conservative | Noisy preferences, want safety |
| 0.5+ | Very conservative | Minimal deviation required |

## PPO Training Loop

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

config = PPOConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    learning_rate=1e-5,
    batch_size=64,
    mini_batch_size=8,
    ppo_epochs=4,
    kl_penalty="kl",               # "kl", "abs", or "mse"
    init_kl_coef=0.2,
    target_kl=6.0,                  # adaptive KL -- increases coef if KL exceeds this
    cliprange=0.2,
    vf_coef=0.1,
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

for batch in dataloader:
    query_tensors = tokenizer(batch["prompt"], return_tensors="pt", padding=True).input_ids
    response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=256)
    response_text = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Score with reward model
    rewards = [reward_model.score(q, r) for q, r in zip(batch["prompt"], response_text)]
    rewards = [torch.tensor(r, dtype=torch.float32) for r in rewards]

    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
```

## KTO / ORPO / SimPO with TRL

```python
# KTO -- binary signal (good/bad), no pairwise data needed
from trl import KTOTrainer, KTOConfig

kto_config = KTOConfig(
    output_dir="kto_output",
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    max_length=1024,
    bf16=True,
    # KTO-specific: ratio of desirable to undesirable examples
    desirable_weight=1.0,
    undesirable_weight=1.0,
)
trainer = KTOTrainer(model=model, config=kto_config, ...)
```

```python
# ORPO -- no reference model, odds ratio preference optimization
from trl import ORPOTrainer, ORPOConfig

orpo_config = ORPOConfig(
    output_dir="orpo_output",
    learning_rate=8e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    max_length=1024,
    bf16=True,
    beta=0.1,  # Weight of the odds ratio loss
)
trainer = ORPOTrainer(model=model, config=orpo_config, ...)
```

```python
# CPO / SimPO -- reference-free, length-normalized
from trl import CPOTrainer, CPOConfig

cpo_config = CPOConfig(
    output_dir="simpo_output",
    learning_rate=5e-7,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    max_length=1024,
    bf16=True,
    loss_type="simpo",       # "simpo" for SimPO, "sigmoid" for standard CPO
    cpo_alpha=1.0,           # NLL loss weight (SimPO uses this)
    simpo_gamma=0.5,         # Reward margin (SimPO-specific)
)
trainer = CPOTrainer(model=model, config=cpo_config, ...)
```

### When to Use Which

| Scenario | Best Method |
|----------|-------------|
| Standard pairwise preferences, proven baseline | DPO |
| Only thumbs up/down data (no pairwise) | KTO |
| Memory-constrained (can't load reference model) | ORPO or SimPO |
| DPO overfitting or reward hacking | SimPO (length normalization helps) |
| Need shaped reward signal | PPO |
| Want simplest possible setup | ORPO |

## Gotchas and Anti-Patterns

### Reward Hacking
- **Symptom**: reward increases but output quality degrades (longer, repetitive, or adversarial text)
- **Fix**: increase KL penalty, add length penalty, use ensemble of reward models
- **Detection**: track reward AND KL divergence; if reward rises while KL explodes, you're hacking

### KL Divergence Tuning
- Monitor `kl_divergence` metric every training run. Healthy range: 0.5-10 nats
- KL > 15: model has drifted too far, outputs may be degenerate
- KL ~ 0: model isn't learning, increase LR or decrease beta

### Reference Model Management
- DPO with LoRA: set `ref_model=None` in TRL -- it uses the frozen base weights automatically
- PPO: reference model must be a separate copy, kept frozen. Don't share weights
- Memory trick: load ref model in 8-bit quantization if GPU-constrained

### Mode Collapse
- **Symptom**: model generates same response structure regardless of prompt
- **Fix**: lower learning rate, increase KL penalty, ensure diverse preference data
- **Prevention**: validate on held-out prompts every N steps; track output diversity metrics (distinct-n, entropy)

### Common Mistakes
- Training reward model for >1 epoch -- overfits fast on preference data
- Using SFT learning rates (2e-5) for DPO/PPO -- too high; use 5e-7 to 5e-6
- Not filtering preference data for ties/ambiguous pairs -- degrades signal
- Forgetting `max_prompt_length` in DPO -- prompts eat into generation budget
- Running PPO without reward model normalization -- unstable training
