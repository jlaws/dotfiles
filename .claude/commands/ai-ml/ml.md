---
name: ml
description: "ML/AI consultation — launches ml-engineer subagent. Use when working with model training, inference pipelines, LLM applications, or ML deployment."
---

Before invoking the subagent, gather diagnostic context:

1. **Detect ML framework** from project config (requirements.txt, pyproject.toml, setup.py) — look for torch, tensorflow, jax, sklearn, transformers, etc.
2. **Identify model artifacts** by searching for .pt, .onnx, .safetensors, .pkl, .h5, saved_model/, checkpoints/, or model config files.
3. **Check ML tooling config** (mlflow, wandb, dvc, hydra configs, experiment tracking setup).
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a component, scope to that; otherwise scan for ML-related directories like models/, training/, data/, pipelines/).

Use the ml-engineer subagent to help with: $ARGUMENTS
