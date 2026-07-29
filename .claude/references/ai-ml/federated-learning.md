# Federated Learning & Privacy-Preserving ML

## Privacy Technique Selection

| Technique | Protects Against | Overhead | Use When |
|-----------|-----------------|----------|----------|
| **Differential Privacy (DP-SGD)** | Membership inference, model inversion | 10-30% accuracy loss | Training on sensitive data |
| **PII Detection/Redaction** | Data leakage in text | Minimal (preprocessing) | Any text pipeline with personal data |
| **Federated Learning** | Raw data exposure | Communication cost | Data cannot leave client devices |
| **Secure Aggregation** | Server seeing individual updates | Crypto overhead | FL with untrusted server |
| **Model Unlearning** | Right to erasure compliance | Retraining cost | GDPR Art. 17 requests |
| **K-Anonymity / L-Diversity** | Re-identification in tabular data | Data utility loss | Publishing/sharing datasets |

**Decision rule**: Start with PII detection (cheap, always useful). Add DP-SGD for provable guarantees. Use FL when data cannot be centralized. Combine for defense in depth.

## Federated Strategy Selection

| Strategy | Privacy | Scale | Non-IID Tolerance | Best For |
|----------|---------|-------|-------------------|----------|
| **FedAvg** | Low | Cross-silo | Low | Homogeneous data, fast prototyping |
| **FedProx** | Low | Cross-silo | Medium | Heterogeneous clients, stragglers |
| **FedAvg + DP** | High | Either | Low | Regulatory compliance |
| **FedSGD + SecAgg** | Very High | Cross-silo | Low | Finance, healthcare |
| **Compressed FedAvg** | Low | Cross-device | Low | Mobile/IoT, bandwidth-constrained |
| **Scaffold** | Low | Cross-silo | High | Highly non-IID data |

### Cross-Device vs Cross-Silo

| Dimension | Cross-Device | Cross-Silo |
|-----------|-------------|------------|
| Clients | Millions of phones/IoT | 2-100 organizations |
| Data per client | Small (KB-MB) | Large (GB-TB) |
| Participation | 0.1-1% per round | 100% per round |
| Trust model | Untrusted | Semi-trusted partners |

## FedAvg Implementation

```python
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Dict, Tuple

class FedAvgServer:
    """Central server for federated averaging."""
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model

    def aggregate(self, client_updates: List[Tuple[Dict, int]]):
        """Weighted average of client models by dataset size."""
        total_samples = sum(n for _, n in client_updates)
        state = self.global_model.state_dict()
        for key in state:
            state[key] = sum(
                s[key].float() * (n / total_samples) for s, n in client_updates)
        self.global_model.load_state_dict(state)

    def run_round(self, clients: List["FedClient"]):
        global_state = deepcopy(self.global_model.state_dict())
        updates = [c.train(global_state) for c in clients]
        self.aggregate(updates)

class FedClient:
    """Client that trains locally and returns model updates."""
    def __init__(self, model_fn, train_loader, lr=0.01, local_epochs=5):
        self.model = model_fn()
        self.train_loader = train_loader
        self.lr, self.local_epochs = lr, local_epochs

    def train(self, global_state: Dict) -> Tuple[Dict, int]:
        self.model.load_state_dict(global_state)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        num_samples = 0
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                optimizer.zero_grad()
                criterion(self.model(x), y).backward()
                optimizer.step()
                num_samples += len(x)
        return self.model.state_dict(), num_samples // self.local_epochs
```

## FedProx: Proximal Term for Heterogeneity

```python
class FedProxClient(FedClient):
    """Adds L2 penalty toward global model to limit client drift."""
    def __init__(self, model_fn, train_loader, lr=0.01, local_epochs=5, mu=0.01):
        super().__init__(model_fn, train_loader, lr, local_epochs)
        self.mu = mu

    def train(self, global_state: Dict) -> Tuple[Dict, int]:
        self.model.load_state_dict(global_state)
        self.model.train()
        global_params = {k: v.clone().detach() for k, v in self.model.named_parameters()}
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        num_samples = 0
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(x), y)
                for name, param in self.model.named_parameters():
                    loss += (self.mu / 2) * ((param - global_params[name]) ** 2).sum()
                loss.backward()
                optimizer.step()
                num_samples += len(x)
        return self.model.state_dict(), num_samples // self.local_epochs
```

## Differential Privacy Integration

DP-SGD clips **per-sample** gradients to `max_grad_norm`, then adds Gaussian noise scaled by
`noise_multiplier * max_grad_norm` to the summed batch gradient.

**Do not hand-roll this.** After `loss.backward()` PyTorch has already reduced gradients across
the batch, so clipping `p.grad` at that point bounds the *batch* norm, not any individual
sample's contribution -- it yields no per-sample sensitivity bound and therefore no
`(epsilon, delta)` guarantee, however plausible the clip-then-noise code looks. Getting the
guarantee requires materializing per-sample gradients (vectorized hooks) plus an accountant to
compose epsilon over steps. Use Opacus, which does both: see
`ai-ml/federated-learning/privacy-techniques.md` for the `make_private_with_epsilon` setup and the
`target_epsilon` / `max_grad_norm` parameter ranges.

### Privacy Budget Intuition
- epsilon < 1: Strong privacy, significant accuracy cost
- epsilon 1-10: Moderate privacy, reasonable utility
- epsilon > 10: Weak privacy, may not provide meaningful protection
- epsilon is cumulative across training; more epochs = more privacy spent

### Privacy Budget Tracking

```python
from dataclasses import dataclass, field

@dataclass
class PrivacyBudget:
    """Track cumulative privacy spend across queries/training runs."""
    total_epsilon: float
    total_delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    history: list[dict] = field(default_factory=list)

    @property
    def remaining_epsilon(self) -> float:
        return self.total_epsilon - self.spent_epsilon

    def can_spend(self, epsilon: float, delta: float) -> bool:
        return (self.spent_epsilon + epsilon <= self.total_epsilon
                and self.spent_delta + delta <= self.total_delta)

    def spend(self, epsilon: float, delta: float, description: str = ""):
        if not self.can_spend(epsilon, delta):
            raise PrivacyBudgetExhausted(
                f"Cannot spend eps={epsilon}, delta={delta}. "
                f"Remaining: eps={self.remaining_epsilon:.2f}")
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.history.append({"epsilon": epsilon, "delta": delta,
                             "description": description})
```

## Communication Efficiency

### Top-K Gradient Sparsification

```python
class TopKCompressor:
    """Keep only top-k% of gradient values; accumulate residuals."""
    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio
        self.residuals = {}

    def compress(self, model: nn.Module) -> Dict:
        compressed = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.data
            if name in self.residuals:
                grad = grad + self.residuals[name]
            flat = grad.view(-1)
            k = max(1, int(len(flat) * self.compress_ratio))
            _, indices = torch.topk(flat.abs(), k)
            values = flat[indices]
            residual = flat.clone()
            residual[indices] = 0
            self.residuals[name] = residual.view_as(grad)
            compressed[name] = (values, indices)
        return compressed
```

## Secure Aggregation Sketch

```python
class SecureAggregator:
    """Masking-based secure aggregation (conceptual)."""
    def generate_masks(self, client_ids: list, param_shape):
        masks = {cid: torch.zeros(param_shape) for cid in client_ids}
        for i, c1 in enumerate(client_ids):
            for c2 in client_ids[i + 1:]:
                g = torch.Generator().manual_seed(hash((c1, c2)) % (2 ** 32))
                mask = torch.randn(param_shape, generator=g)
                masks[c1] += mask; masks[c2] -= mask  # cancels on sum
        return masks
```

**Production options**: TensorFlow Federated (built-in SecAgg), PySyft (MPC support), Flower (pluggable aggregation).

## Compliance Mapping

| Requirement | GDPR | CCPA | HIPAA | Technique |
|------------|------|------|-------|-----------|
| Right to erasure | Art. 17 | Sec. 1798.105 | -- | Model unlearning |
| Data minimization | Art. 5(1)(c) | -- | Min. Necessary | PII redaction, DP |
| Purpose limitation | Art. 5(1)(b) | -- | -- | Access controls, audit logs |
| Automated decisions | Art. 22 | -- | -- | Explainability, human review |
| De-identification | Recital 26 | Sec. 1798.140(o) | Safe Harbor | K-anonymity, DP |
| Breach notification | Art. 33 | Sec. 1798.150 | Breach Rule | Encryption, access logs |

## Gotchas

- **Non-IID data kills convergence**: FedAvg diverges with skewed labels; use FedProx, Scaffold, or data sharing
- **Privacy budget exhaustion**: Each round consumes epsilon; track cumulative budget with RDP (use Opacus)
- **DP-SGD and BatchNorm**: Opacus does not support BatchNorm (tracks per-sample stats). Replace with GroupNorm/LayerNorm; use `ModuleValidator.fix(model)`
- **Epsilon composition**: Multiple queries/runs compound epsilon. Use Renyi DP for tighter bounds
- **Weight divergence**: Too many local epochs causes drift; reduce epochs or increase mu in FedProx
- **Communication bottleneck**: Model size x clients x rounds; compress aggressively for cross-device
- **Secure aggregation dropout**: Dropped clients break mask cancellation; need threshold secret sharing
- **Model poisoning**: Malicious clients send adversarial updates; use robust aggregation (trimmed mean, Krum)
- **PII detection false negatives**: Presidio catches common patterns but misses context-dependent PII; layer regex + NER + LLM detection
- **Model inversion attacks**: Attackers reconstruct training data from outputs; DP-SGD protects; limit API output to class labels
- **Compliance is not just technical**: Also need data processing agreements, DPIAs, consent management, audit trails

## References

Extended code examples in `ai-ml/federated-learning/privacy-techniques.md`:
- DP-SGD with Opacus (full setup + parameters)
- PII detection with Presidio (basic + custom recognizers + pipeline integration)
- Model unlearning (exact + SISA)
- Flower framework pattern
