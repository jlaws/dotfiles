---
type: llm
focus: { source: file, path: serving-plan.md }
---

All claims are required.
PASS if serving-plan.md computes KV cache per token as 2 x 32 x 8 x 128 x 2 bytes, which is 131,072 bytes (128 KB).
PASS if serving-plan.md totals the requirement at about 32 GB (64 x 4,096 x 128 KB) and concludes it fits alongside the 16 GB of weights.
PASS if serving-plan.md sets gpu_memory_utilization at 0.85 to 0.90 (not 0.95) and sets max_model_len to 4,096 or the real maximum instead of the model's theoretical maximum.
PASS if serving-plan.md relies on continuous batching (vLLM default) and names at least one latency setting such as CUDA graphs, prefix caching, or quantization.
FAIL if serving-plan.md launches with gpu_memory_utilization of 0.95 or higher.
