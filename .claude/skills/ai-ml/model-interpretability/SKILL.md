---
name: model-interpretability
description: Interpret ML model predictions using SHAP, LIME, attention visualization, and probing techniques. Use when explaining model decisions, debugging model behavior, or building trust in ML systems.
---

# Model Interpretability

Master interpretability techniques to understand and explain ML model behavior, from feature attribution to mechanistic analysis.

## When to Use This Skill

- Explaining individual predictions to stakeholders
- Debugging unexpected model behavior
- Understanding what features drive predictions
- Validating model reasoning aligns with domain knowledge
- Meeting regulatory requirements for explainability
- Discovering spurious correlations or shortcuts

## SHAP (SHapley Additive exPlanations)

### TreeSHAP for Tree Models
```python
import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_tree_model(model, X_train, X_explain):
    """Explain tree-based model predictions with SHAP."""

    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_explain)

    return explainer, shap_values


def visualize_shap(explainer, shap_values, X_explain, feature_names=None):
    """Create SHAP visualizations."""

    # Summary plot (feature importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()

    # Dependence plot for top feature
    shap.dependence_plot(0, shap_values, X_explain, feature_names=feature_names)

    # Force plot for single prediction
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_explain[0],
        feature_names=feature_names,
        matplotlib=True,
    )

    # Waterfall plot
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_explain[0],
        feature_names=feature_names,
    ))
```

### DeepSHAP for Neural Networks
```python
import shap
import torch

def explain_neural_network(model, X_background, X_explain):
    """Explain neural network predictions with DeepSHAP."""

    # Background data for expected value estimation
    background = torch.tensor(X_background[:100], dtype=torch.float32)

    # Create DeepExplainer
    explainer = shap.DeepExplainer(model, background)

    # Calculate SHAP values
    shap_values = explainer.shap_values(
        torch.tensor(X_explain, dtype=torch.float32)
    )

    return explainer, shap_values


# For image models
def explain_image_model(model, images, class_idx=None):
    """Explain image classification predictions."""

    # Use GradientExplainer for efficiency
    explainer = shap.GradientExplainer(model, images[:50])

    # Explain specific images
    shap_values = explainer.shap_values(images)

    # Visualize
    shap.image_plot(shap_values, images)

    return shap_values
```

## LIME (Local Interpretable Model-agnostic Explanations)

### Tabular Data
```python
from lime import lime_tabular

def explain_with_lime(model, X_train, X_explain, feature_names, class_names=None):
    """Explain predictions with LIME."""

    # Create explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        random_state=42,
    )

    # Explain single prediction
    explanation = explainer.explain_instance(
        X_explain[0],
        model.predict_proba,
        num_features=10,
        num_samples=5000,
    )

    # Get feature contributions
    contributions = explanation.as_list()
    print("Feature contributions:")
    for feature, weight in contributions:
        print(f"  {feature}: {weight:.4f}")

    # Visualize
    explanation.show_in_notebook()
    # Or save as HTML
    explanation.save_to_file("lime_explanation.html")

    return explanation
```

### Text Data
```python
from lime.lime_text import LimeTextExplainer

def explain_text_prediction(model, text, class_names):
    """Explain text classification predictions."""

    explainer = LimeTextExplainer(class_names=class_names)

    # Predict function that takes list of strings
    def predict_fn(texts):
        # Tokenize and predict
        return model.predict_proba(texts)

    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=10,
        num_samples=1000,
    )

    # Highlight important words
    explanation.show_in_notebook(text=True)

    return explanation
```

### Image Data
```python
from lime import lime_image
from skimage.segmentation import mark_boundaries

def explain_image_prediction(model, image, class_idx):
    """Explain image classification predictions."""

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image,
        model.predict,
        top_labels=5,
        hide_color=0,
        num_samples=1000,
    )

    # Get mask for predicted class
    temp, mask = explanation.get_image_and_mask(
        class_idx,
        positive_only=True,
        num_features=5,
        hide_rest=False,
    )

    # Visualize
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title(f"LIME explanation for class {class_idx}")
    plt.axis('off')

    return explanation
```

## Attention Visualization

### Transformer Attention
```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens, layer=0, head=0):
    """Visualize attention weights from transformer."""

    # attention_weights: (layers, heads, seq_len, seq_len)
    attn = attention_weights[layer, head].detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        annot=False,
    )
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.title(f"Attention weights (Layer {layer}, Head {head})")
    plt.tight_layout()

    return attn


def get_attention_weights(model, input_ids, attention_mask=None):
    """Extract attention weights from transformer model."""

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    # Stack all layers: (num_layers, batch, heads, seq, seq)
    attentions = torch.stack(outputs.attentions)

    # Average over heads if desired
    # avg_attention = attentions.mean(dim=2)

    return attentions


def attention_rollout(attentions, add_residual=True):
    """
    Compute attention rollout for global token importance.

    Attentions: (layers, batch, heads, seq, seq)
    """
    # Average over heads
    attn = attentions.mean(dim=2)  # (layers, batch, seq, seq)

    # Add residual connection
    if add_residual:
        eye = torch.eye(attn.size(-1), device=attn.device)
        attn = 0.5 * attn + 0.5 * eye

    # Rollout through layers
    rollout = attn[0]
    for layer_attn in attn[1:]:
        rollout = torch.matmul(layer_attn, rollout)

    # Normalize
    rollout = rollout / rollout.sum(dim=-1, keepdim=True)

    return rollout
```

### Attention Caveats
```python
"""
IMPORTANT: Attention weights are NOT explanations!

Caveats:
1. Attention != importance - tokens can be important without high attention
2. Attention patterns vary significantly across heads/layers
3. Raw attention ignores value vectors and downstream processing
4. Gradient-based methods often more faithful

Use attention visualization for:
- Understanding information flow
- Debugging attention patterns
- Qualitative analysis

DON'T use for:
- Feature importance claims
- Regulatory explanations
- Definitive causal claims
"""

# Better: Gradient-based attention attribution
def gradient_attention_attribution(model, input_ids, target_idx):
    """More faithful attention attribution using gradients."""

    model.eval()
    input_ids.requires_grad = False

    # Get attention weights with gradients
    outputs = model(input_ids, output_attentions=True)

    # Compute gradient of target logit w.r.t. attention
    target_logit = outputs.logits[0, target_idx]

    # Get gradients
    attention_grads = torch.autograd.grad(
        target_logit,
        outputs.attentions,
        retain_graph=True,
    )

    # Weight attention by gradients
    weighted_attention = []
    for attn, grad in zip(outputs.attentions, attention_grads):
        weighted = attn * grad
        weighted_attention.append(weighted)

    return weighted_attention
```

## Integrated Gradients

```python
import torch
from typing import Callable

def integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor = None,
    steps: int = 50,
) -> torch.Tensor:
    """
    Compute integrated gradients attribution.

    Args:
        model: Neural network model
        input_tensor: Input to explain (1, ...)
        target_class: Class index for attribution
        baseline: Reference input (default: zeros)
        steps: Number of interpolation steps

    Returns:
        Attribution scores same shape as input
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Generate interpolated inputs
    alphas = torch.linspace(0, 1, steps, device=input_tensor.device)
    interpolated = baseline + alphas.view(-1, *[1] * (input_tensor.dim() - 1)) * (input_tensor - baseline)

    # Compute gradients
    interpolated.requires_grad_(True)
    outputs = model(interpolated)
    target_outputs = outputs[:, target_class]

    gradients = torch.autograd.grad(
        outputs=target_outputs.sum(),
        inputs=interpolated,
    )[0]

    # Average gradients and scale by input difference
    avg_gradients = gradients.mean(dim=0)
    integrated_grads = (input_tensor - baseline) * avg_gradients

    return integrated_grads


# Usage for image model
def explain_image_with_ig(model, image, target_class):
    """Explain image prediction with Integrated Gradients."""

    attributions = integrated_gradients(
        model,
        image.unsqueeze(0),
        target_class,
        steps=100,
    )

    # Visualize
    attr_viz = attributions.squeeze().permute(1, 2, 0).abs().sum(dim=-1)
    attr_viz = (attr_viz - attr_viz.min()) / (attr_viz.max() - attr_viz.min())

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title("Original")
    plt.subplot(1, 3, 2)
    plt.imshow(attr_viz, cmap='hot')
    plt.title("Attribution")
    plt.subplot(1, 3, 3)
    plt.imshow(image.permute(1, 2, 0))
    plt.imshow(attr_viz, cmap='hot', alpha=0.5)
    plt.title("Overlay")
    plt.tight_layout()

    return attributions
```

## Probing Classifiers

```python
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class ProbingClassifier:
    """Probe neural network representations for specific properties."""

    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.activations = []
        self._register_hook()

    def _register_hook(self):
        """Register forward hook to capture activations."""
        def hook(module, input, output):
            self.activations.append(output.detach().cpu())

        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook)
                return

        raise ValueError(f"Layer {self.layer_name} not found")

    def extract_representations(self, dataloader) -> torch.Tensor:
        """Extract representations from the target layer."""
        self.activations = []
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                self.model(batch["input"])

        return torch.cat(self.activations, dim=0)

    def probe(self, representations: torch.Tensor, labels: torch.Tensor) -> dict:
        """Train and evaluate probing classifier."""

        # Flatten representations
        X = representations.view(representations.size(0), -1).numpy()
        y = labels.numpy()

        # Train simple linear probe
        probe = LogisticRegression(max_iter=1000, random_state=42)

        # Cross-validation
        scores = cross_val_score(probe, X, y, cv=5, scoring='accuracy')

        return {
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
            "scores": scores,
        }


def layer_wise_probing(model, dataloader, labels, property_name):
    """Probe all layers to find where information is encoded."""

    results = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.TransformerEncoderLayer)):
            try:
                prober = ProbingClassifier(model, name)
                representations = prober.extract_representations(dataloader)
                result = prober.probe(representations, labels)
                results[name] = result
                print(f"{name}: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}")
            except Exception as e:
                print(f"{name}: Failed - {e}")

    return results
```

## Captum Library

```python
from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    GradientShap,
    DeepLift,
    Saliency,
    InputXGradient,
    LayerConductance,
)

def comprehensive_attribution(model, input_tensor, target):
    """Compare multiple attribution methods."""

    results = {}

    # Integrated Gradients
    ig = IntegratedGradients(model)
    results["integrated_gradients"] = ig.attribute(input_tensor, target=target)

    # Gradient SHAP
    gs = GradientShap(model)
    baseline = torch.zeros_like(input_tensor)
    results["gradient_shap"] = gs.attribute(
        input_tensor, baselines=baseline, target=target
    )

    # DeepLift
    dl = DeepLift(model)
    results["deeplift"] = dl.attribute(input_tensor, target=target)

    # Saliency (simple gradient)
    sal = Saliency(model)
    results["saliency"] = sal.attribute(input_tensor, target=target)

    return results
```

## Best Practices

1. **Use multiple methods**: No single method is always best
2. **Validate explanations**: Check if they align with domain knowledge
3. **Be careful with attention**: It's not a faithful explanation
4. **Test faithfulness**: Perturb important features and check predictions change
5. **Consider the audience**: Technical vs non-technical explanations
6. **Document limitations**: Be honest about what explanations can/cannot tell us

## Common Pitfalls

- **Trusting a single method**: Different methods highlight different aspects
- **Ignoring baseline choice**: SHAP/IG results depend heavily on baseline
- **Attention as explanation**: Attention weights are not importances
- **Post-hoc rationalization**: Explanations may not reflect true reasoning
- **Ignoring uncertainty**: Point estimates don't capture explanation variance
