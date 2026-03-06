# LLM Training Pipeline

## Training Method Decision Table

| Method | VRAM (7B) | Quality vs Full FT | When to Use |
|--------|-----------|---------------------|-------------|
| Full pretraining | Cluster-scale | N/A (creates base) | New architecture, proprietary data, custom tokenizer |
| Continued pretraining | 60-80 GB | Domain adaptation | Inject domain knowledge before fine-tuning |
| Full fine-tuning | 60-80 GB | Baseline | Unlimited compute, max quality, own the weights |
| LoRA (r=64) | 18-24 GB | 95-99% | Production adapters, multi-tenant serving |
| QLoRA (4-bit + LoRA) | 6-10 GB | 90-97% | Single GPU, prototyping, budget-constrained |
| LoRA (r=8-16) | 14-18 GB | 90-95% | Quick experiments, narrow domain tasks |

**Decision rule**: Start with QLoRA to validate the task is learnable, then LoRA r=64 or full FT for production.

## Pretraining Scale Planning

| Scale (params) | Data Budget | Batch Size (tokens) | LR | Key Considerations |
|---|---|---|---|---|
| 100M-500M | 5-10B tokens | 256K-512K | 3e-4 | Quick experiments, single-node |
| 1-3B | 20-60B tokens | 512K-1M | 2e-4 | Chinchilla ~20 tok/param |
| 7-13B | 140-500B tokens | 1M-2M | 1.5e-4 | Multi-node, activation ckpt |
| 30-70B | 1-2T tokens | 2M-4M | 1e-4 | TP+PP, quality annealing phase |
| 200B+ | 4T+ tokens | 4M-8M | 6e-5 | MoE likely, multi-phase curriculum |

```python
import math

def chinchilla_optimal(compute_flops: float) -> dict:
    """Estimate optimal N (params) and D (tokens) for given compute."""
    n_opt = (compute_flops / 120) ** 0.5
    d_opt = 20 * n_opt
    return {"params_B": n_opt / 1e9, "tokens_T": d_opt / 1e12, "compute_flops": compute_flops}

def estimate_training_time(params_B: float, tokens_T: float, num_gpus: int,
                           gpu_tflops: float = 312, mfu: float = 0.40) -> dict:
    flops_total = 6 * params_B * 1e9 * tokens_T * 1e12
    effective_tflops = gpu_tflops * mfu * num_gpus
    seconds = flops_total / (effective_tflops * 1e12)
    return {"gpu_hours": seconds * num_gpus / 3600, "wall_days": seconds / 86400,
            "total_pflops_days": flops_total / (1e15 * 86400), "mfu": mfu}
```

For data mixing, tokenizer co-design, checkpoint management, and stability monitoring patterns, see `references/pretraining-patterns.md`.

## Dataset Formatting

### Chat Template Format (Preferred)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def format_chat(example):
    """Use the model's native chat template. Always prefer this for instruct models."""
    messages = [
        {"role": "system", "content": example.get("system", "You are a helpful assistant.")},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
```

### Dataset Prep Pipeline

```python
from datasets import load_dataset, DatasetDict

def prepare_dataset(path, tokenizer, max_length=2048, test_size=0.05):
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.map(format_chat, remove_columns=ds.column_names)
    # Filter overlength samples rather than truncating -- avoids training on garbage
    ds = ds.filter(lambda x: len(tokenizer.encode(x["text"])) <= max_length)
    split = ds.train_test_split(test_size=test_size, seed=42)
    return DatasetDict({"train": split["train"], "test": split["test"]})
```

**Gotcha**: Truncating mid-response teaches the model to produce incomplete outputs. Filter or increase `max_length` instead.

## PEFT / LoRA Configuration

```python
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,           # alpha = 2*r is a solid default
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Expect 1-3% of total
```

### Target Module Selection

| Model Family | Recommended Targets | Notes |
|-------------|-------------------|-------|
| Llama/Mistral | `q,k,v,o_proj` + `gate,up,down_proj` | All linear layers for best quality |
| GPT-NeoX/Pythia | `query_key_value`, `dense` | Fused QKV attention |
| Phi | `q_proj,k_proj,v_proj,dense` | Check model config for names |

Use `model.named_modules()` to discover actual layer names if unsure.

## QLoRA Setup

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,       # Nested quantization saves ~0.4 GB/B params
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

**Gotcha**: `use_reentrant=False` is mandatory with LoRA + gradient checkpointing. The default (`True`) silently skips gradients for LoRA params.

## Training with SFTTrainer

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,        # Effective batch = 16
    learning_rate=2e-4,                   # LoRA tolerates higher LR than full FT
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    max_seq_length=2048,
    dataset_text_field="text",
    packing=True,                         # Pack short sequences together for efficiency
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
trainer = SFTTrainer(
    model=model, args=training_args,
    train_dataset=dataset["train"], eval_dataset=dataset["test"],
    peft_config=lora_config, tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("./final_adapter")
```

## Adapter Merging

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./final_adapter")
merged = model.merge_and_unload()
merged.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

**Gotcha**: Don't merge QLoRA adapters directly onto the quantized base. Load the base in full precision (bf16/fp16) first, then load the adapter, then merge.

## Gotchas and Anti-Patterns

### Fine-Tuning
- **Chat template mismatch**: training with one template, inferring with another, destroys quality. Save tokenizer alongside adapter; use `apply_chat_template` consistently
- **Padding direction**: training = `"right"`, batch inference = `"left"`. Forgetting to switch is a silent quality killer
- Setting `lora_alpha = r` instead of `2*r` -- underscales adapter contribution
- Using `packing=True` without many short examples -- wastes compute if examples are near `max_seq_length`
- Not setting `pad_token` -- many models (Llama) lack one: `tokenizer.pad_token = tokenizer.eos_token`
- Training on prompt tokens -- use `DataCollatorForCompletionOnly` or mask labels manually
- Evaluating with `do_sample=True` -- introduces variance making comparison meaningless

### Pretraining
- **Data dedup across phases**: repeated high-quality data in annealing causes memorization; track doc-level overlap
- **Tokenizer fertility drift**: adding domains with poor coverage drops effective sequence length
- **Chinchilla is a lower bound**: overtrained models transfer better; plan 2-5x Chinchilla tokens at smaller scales
- **Batch size ramp**: ramp too fast = instability; ramp over 2-5% of total steps
- **Checkpoint storage**: 70B checkpoint ~140GB; with Adam states ~420GB per save; budget 10-50TB
- Loss spikes below 1B tokens are normal -- don't rollback before 0.5-1% of total training tokens
- MFU drops with PP: expect 30-35% with pipeline parallelism vs 40-45% with pure TP/FSDP
- LR restarts between phases: use smooth transitions, not cold restarts
- bf16 vs fp32 accumulation: always accumulate gradients in fp32; skipping causes slow divergence
- Z-loss regularization: add small aux loss on logit magnitude (1e-4) to prevent logit drift
