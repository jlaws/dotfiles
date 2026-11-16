---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:ml-engineer agent to plan serving a Llama-style 8B model with vLLM on one 80 GB GPU. The model has 32 layers, 8 KV heads, and head dimension 128, runs in bf16, and the weights take about 16 GB. Requirements: 64 concurrent sequences, each up to 4,096 tokens. Save the plan to ./serving-plan.md. It must give the KV cache bytes per token with the formula, the total KV cache for the requirement, whether it fits, the gpu_memory_utilization and max_model_len values you would launch with and why, and the batching and latency settings you would enable.
