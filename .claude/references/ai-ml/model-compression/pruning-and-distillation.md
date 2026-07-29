# Pruning & Distillation Reference

## Transformer Head Pruning

```python
import torch
import torch.nn as nn

def prune_transformer_heads(model, heads_to_prune):
    """Prune attention heads from a transformer.

    Args:
        heads_to_prune: dict of {layer_idx: [head_indices]}
            e.g., {0: [0, 3], 2: [1, 5, 7]}
    """
    for layer_idx, heads in heads_to_prune.items():
        layer = model.encoder.layer[layer_idx]
        attention = layer.attention.self

        num_heads = attention.num_attention_heads
        head_size = attention.attention_head_size

        keep_heads = sorted(set(range(num_heads)) - set(heads))
        keep_indices = torch.cat([
            torch.arange(h * head_size, (h + 1) * head_size) for h in keep_heads
        ])

        for proj in [attention.query, attention.key, attention.value]:
            proj.weight = nn.Parameter(proj.weight.index_select(0, keep_indices))
            proj.bias = nn.Parameter(proj.bias.index_select(0, keep_indices))

        attention.output.dense.weight = nn.Parameter(
            attention.output.dense.weight.index_select(1, keep_indices)
        )
        attention.num_attention_heads = len(keep_heads)
        attention.all_head_size = len(keep_heads) * head_size

    return model
```

## Knowledge Distillation

### Loss Function

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5):
    """Combined hard-label and soft-label distillation loss."""
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

The `temperature ** 2` rescaling is the detail most implementations drop: softening the
distributions shrinks the KL gradients by roughly `1 / temperature ** 2`, so without it the
soft term silently stops contributing as temperature rises.

### Training Loop

Standard supervised loop with the teacher in `eval()` under `torch.no_grad()`, the student in
`train()`, and gradient clipping on the student's parameters.
