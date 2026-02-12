---
name: llmops-production-monitoring
description: Monitor LLM applications with prompt logging, cost tracking, and guardrail pipelines
---

# LLMOps Production Monitoring

## Monitoring Approach Selection

| Concern | Tool/Approach | When to Use |
|---------|---------------|-------------|
| **Prompt/response logging** | Structured logger + object store | Always; foundation for all other monitoring |
| **Cost tracking** | Token counter + price table | Always; prevents budget blowouts |
| **Latency monitoring** | Histogram metrics (P50/P95/P99) | Always; SLA compliance |
| **Output quality drift** | Embedding distance over time | After stable baseline established |
| **Safety/guardrails** | Input/output classifier pipeline | User-facing applications |
| **Prompt A/B testing** | Feature flag + metric comparison | Optimizing prompt performance |

**Default stack:** Structured logging (all calls) + Prometheus histograms (latency/tokens) + async guardrail pipeline (user-facing).

## Structured Prompt Logging

```python
import uuid, time, json, logging, hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

logger = logging.getLogger("llmops")

@dataclass
class LLMCallRecord:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: str = ""
    prompt_template: str = ""               # Template name/version, not raw prompt
    prompt_hash: str = ""                   # SHA256 of rendered prompt for dedup
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    status: str = "success"                 # success | error | timeout | filtered
    error_message: str = ""
    user_id: str = ""
    experiment_id: str = ""
    metadata: dict = field(default_factory=dict)

    def log(self):
        logger.info(json.dumps(asdict(self)))

def logged_llm_call(client, messages: list, model: str, **kwargs) -> tuple:
    """Wraps any OpenAI-compatible client with structured logging."""
    record = LLMCallRecord(
        model=model,
        prompt_hash=hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:16],
    )
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        record.input_tokens = response.usage.prompt_tokens
        record.output_tokens = response.usage.completion_tokens
        record.latency_ms = (time.perf_counter() - start) * 1000
        return response.choices[0].message.content, record
    except Exception as e:
        record.status = "error"
        record.error_message = str(e)[:500]
        record.latency_ms = (time.perf_counter() - start) * 1000
        raise
    finally:
        record.log()
```

## Token Counting and Cost Tracking

```python
import tiktoken

# Price per 1M tokens (input, output) -- update as pricing changes
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60),
    "claude-3-opus": (15.00, 75.00), "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
}

@dataclass
class CostTracker:
    total_cost_usd: float = 0.0
    calls_by_model: dict = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        in_price, out_price = PRICE_TABLE.get(model, (0.0, 0.0))
        cost = (input_tokens * in_price + output_tokens * out_price) / 1_000_000
        self.total_cost_usd += cost
        entry = self.calls_by_model.setdefault(model, {"calls": 0, "cost": 0.0})
        entry["calls"] += 1; entry["cost"] += cost
        return cost

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try: enc = tiktoken.encoding_for_model(model)
    except KeyError: enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
```

## Latency Monitoring

```python
from prometheus_client import Histogram, Counter, Gauge

LLM_LATENCY = Histogram("llm_call_duration_seconds", "LLM call latency",
    ["model", "prompt_template"], buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60])
LLM_TOKENS = Histogram("llm_tokens_total", "Tokens per call",
    ["model", "direction"], buckets=[50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000])
LLM_ERRORS = Counter("llm_errors_total", "LLM call errors", ["model", "error_type"])

def observe_call(record: LLMCallRecord):
    LLM_LATENCY.labels(model=record.model, prompt_template=record.prompt_template).observe(record.latency_ms / 1000)
    LLM_TOKENS.labels(model=record.model, direction="input").observe(record.input_tokens)
    LLM_TOKENS.labels(model=record.model, direction="output").observe(record.output_tokens)
    if record.status != "success":
        LLM_ERRORS.labels(model=record.model, error_type=record.status).inc()
```

## Output Drift Detection

```python
import numpy as np
from collections import deque
from sentence_transformers import SentenceTransformer

class DriftDetector:
    """Detect output distribution shift using embedding distance over time."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", window: int = 1000):
        self.encoder = SentenceTransformer(model_name)
        self.baseline_centroid: np.ndarray | None = None
        self.recent: deque = deque(maxlen=window)

    def set_baseline(self, texts: list[str]):
        self.baseline_centroid = self.encoder.encode(texts).mean(axis=0)

    def _cosine_dist(self, a, b) -> float:
        return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def record(self, text: str) -> float:
        emb = self.encoder.encode([text])[0]
        self.recent.append(emb)
        return self._cosine_dist(emb, self.baseline_centroid)

    def check_drift(self, threshold: float = 0.3) -> dict:
        if not self.recent or self.baseline_centroid is None:
            return {"drifted": False, "distance": 0.0}
        dist = self._cosine_dist(np.mean(list(self.recent), axis=0), self.baseline_centroid)
        return {"drifted": dist > threshold, "distance": round(dist, 4)}
```

## Guardrail Pipeline

```python
import re
from enum import Enum

class Action(Enum):
    PASS = "pass"
    BLOCK = "block"
    REDACT = "redact"

@dataclass
class GuardrailResult:
    action: Action
    reason: str = ""
    modified_text: str = ""

class GuardrailPipeline:
    def __init__(self):
        self.input_guards, self.output_guards = [], []

    def check_input(self, text: str) -> GuardrailResult:
        for guard in self.input_guards:
            result = guard(text)
            if result.action != Action.PASS:
                return result
        return GuardrailResult(action=Action.PASS)

    def check_output(self, text: str) -> GuardrailResult:
        for guard in self.output_guards:
            result = guard(text)
            if result.action == Action.BLOCK:
                return result
            if result.action == Action.REDACT:
                text = result.modified_text
        return GuardrailResult(action=Action.PASS, modified_text=text)

PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b", "email": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
}

def pii_scrubber(text: str) -> GuardrailResult:
    modified = text
    for pii_type, pattern in PII_PATTERNS.items():
        modified = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", modified)
    if modified != text:
        return GuardrailResult(action=Action.REDACT, reason="PII detected", modified_text=modified)
    return GuardrailResult(action=Action.PASS)

INJECTION_PATTERNS = [r"ignore (all )?(previous|above|prior) instructions",
    r"you are now", r"system prompt", r"reveal your (instructions|prompt|system)"]

def injection_detector(text: str) -> GuardrailResult:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text.lower()):
            return GuardrailResult(action=Action.BLOCK, reason=f"Injection: {pattern}")
    return GuardrailResult(action=Action.PASS)
```

## A/B Testing Prompt Variants

```python
@dataclass
class PromptVariant:
    name: str
    template: str
    weight: float = 0.5

class PromptExperiment:
    """Deterministic hash-based variant assignment for consistent user experience."""
    def __init__(self, experiment_id: str, variants: list[PromptVariant]):
        self.experiment_id, self.variants = experiment_id, variants
        total = sum(v.weight for v in variants)
        cum, self.thresholds = 0.0, []
        for v in variants:
            cum += v.weight / total
            self.thresholds.append(cum)

    def assign(self, user_id: str) -> PromptVariant:
        h = hashlib.sha256(f"{self.experiment_id}:{user_id}".encode()).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        for threshold, variant in zip(self.thresholds, self.variants):
            if bucket <= threshold:
                return variant
        return self.variants[-1]
```

## Gotchas

- **Log prompts, not in hot path**: Write logs asynchronously (queue + background worker); synchronous logging adds 5-20ms per call
- **Token count != cost**: Cached tokens, batch API discounts, and prompt caching change effective pricing -- track actual billed amounts
- **Drift detection cold start**: Need 500+ baseline samples for stable centroid; small baselines produce false positives
- **Guardrail ordering matters**: Run injection detection before PII scrubbing; blocked requests should not be partially processed
- **A/B test duration**: LLM output variance is high; need 1000+ samples per variant for statistical significance
- **tiktoken model coverage**: Not all models have encodings; `cl100k_base` is a reasonable fallback but may miscount by 5-10%
- **Prometheus cardinality**: Do not use prompt_hash as a label; high-cardinality labels kill Prometheus -- aggregate by template name
