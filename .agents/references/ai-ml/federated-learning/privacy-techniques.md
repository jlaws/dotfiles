# Privacy Techniques Reference

## DP-SGD with Opacus

Opacus adds differential privacy to PyTorch training by clipping per-sample gradients and adding calibrated noise.

```python
import torch
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
train_loader = DataLoader(train_dataset, batch_size=256)

# Attach privacy engine
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=10,
    target_epsilon=8.0,       # Privacy budget
    target_delta=1e-5,        # Should be < 1/N (N = dataset size)
    max_grad_norm=1.0,        # Per-sample gradient clipping bound
)

# Training loop is unchanged
for epoch in range(10):
    for batch, targets in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

    # Check privacy spent so far
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Epoch {epoch}: epsilon = {epsilon:.2f}")
```

### Opacus Key Parameters

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| `target_epsilon` | 1-10 | Lower = more private, worse accuracy |
| `target_delta` | 1/N to 1/(10*N) | Probability of privacy failure |
| `max_grad_norm` | 0.1-5.0 | Gradient clipping bound; too low = underfitting |
| `batch_size` | 256-4096 | Larger = better privacy/utility tradeoff |

### Checking Model Compatibility

```python
from opacus.validators import ModuleValidator

errors = ModuleValidator.validate(model, strict=False)
if errors:
    print("Incompatible modules:", errors)
    model = ModuleValidator.fix(model)  # Auto-fix common issues
    # Replaces BatchNorm with GroupNorm, etc.
```

## PII Detection with Presidio

```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

### Basic PII Detection

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

text = "John Smith's SSN is 123-45-6789 and email is john@example.com"

# Detect PII
results = analyzer.analyze(
    text=text,
    language="en",
    entities=["PERSON", "EMAIL_ADDRESS", "US_SSN", "PHONE_NUMBER", "CREDIT_CARD"],
)

for result in results:
    print(f"  {result.entity_type}: '{text[result.start:result.end]}' (score: {result.score:.2f})")

# Anonymize
anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
print(anonymized.text)
# "<PERSON>'s SSN is <US_SSN> and email is <EMAIL_ADDRESS>"
```

### Custom PII Recognizer

```python
from presidio_analyzer import PatternRecognizer, Pattern

# Detect internal employee IDs (e.g., EMP-12345)
employee_id_recognizer = PatternRecognizer(
    supported_entity="EMPLOYEE_ID",
    patterns=[Pattern(name="emp_id", regex=r"EMP-\d{5}", score=0.9)],
)

analyzer.registry.add_recognizer(employee_id_recognizer)
```

### PII in ML Pipelines

```python
def sanitize_training_data(texts: list[str]) -> list[str]:
    """Remove PII from training data before model training."""
    sanitized = []
    for text in texts:
        results = analyzer.analyze(text=text, language="en")
        if results:
            anon = anonymizer.anonymize(text=text, analyzer_results=results)
            sanitized.append(anon.text)
        else:
            sanitized.append(text)
    return sanitized

# Apply before training
clean_texts = sanitize_training_data(raw_texts)
```

## Model Unlearning

When a user requests data deletion (GDPR Art. 17), you must ensure their data doesn't influence the model.

### Exact Unlearning

```python
def exact_unlearn(model_class, full_dataset, remove_indices: set, train_fn):
    """Retrain from scratch without the removed data. Gold standard but expensive."""
    remaining = [d for i, d in enumerate(full_dataset) if i not in remove_indices]
    new_model = model_class()
    train_fn(new_model, remaining)
    return new_model
```

### SISA (Sharded, Isolated, Sliced, Aggregated)

```python
def sisa_train(model_class, dataset, n_shards: int = 5, train_fn=None):
    """Train separate models on data shards. Unlearning only retrains affected shard."""
    shards = [dataset[i::n_shards] for i in range(n_shards)]
    models = []
    for shard in shards:
        m = model_class()
        train_fn(m, shard)
        models.append(m)
    return models, shards

def sisa_unlearn(models, shards, remove_idx: int, model_class, train_fn):
    """Only retrain the shard containing the removed data point."""
    shard_idx = remove_idx % len(shards)
    shards[shard_idx] = [d for d in shards[shard_idx] if d["id"] != remove_idx]
    models[shard_idx] = model_class()
    train_fn(models[shard_idx], shards[shard_idx])
    return models, shards

def sisa_predict(models, x):
    """Ensemble prediction across shards."""
    predictions = [m(x) for m in models]
    return sum(predictions) / len(predictions)
```

## Flower Framework Pattern

```python
import flwr as fl

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, lr=0.01):
        self.model, self.train_loader, self.val_loader, self.lr = (
            model, train_loader, val_loader, lr)

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        sd = dict(zip(self.model.state_dict().keys(), [torch.tensor(v) for v in params]))
        self.model.load_state_dict(sd)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.model.train()
        for x, y in self.train_loader:
            opt.zero_grad(); nn.CrossEntropyLoss()(self.model(x), y).backward(); opt.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.val_loader:
                correct += (self.model(x).argmax(1) == y).sum().item()
                total += len(y)
        return float(1 - correct / total), total, {"accuracy": correct / total}
```
