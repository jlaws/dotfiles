# Retry and Fallback Strategies

Provider fallback and retry patterns for structured LLM output extraction. For method selection, see the [Structured Output section](../SKILL.md#structured-output) in the parent skill.

## Multi-Provider Fallback

```python
import time
from pydantic import ValidationError

def extract_with_fallback(
    text: str,
    schema_cls,
    max_retries: int = 3,
) -> dict | None:
    """Try OpenAI first, fall back to Anthropic, then return None."""
    providers = [
        ("openai", lambda: extract_openai(text, schema_cls)),
        ("anthropic", lambda: extract_anthropic(text, schema_cls)),
    ]

    for provider_name, extract_fn in providers:
        for attempt in range(max_retries):
            try:
                result = extract_fn()
                return result.model_dump()
            except ValidationError as e:
                print(f"{provider_name} attempt {attempt+1} validation error: {e}")
                time.sleep(0.5 * (attempt + 1))
            except Exception as e:
                print(f"{provider_name} failed: {e}")
                break  # Try next provider
    return None
```

## Retry Strategy Selection

| Strategy | When to Use | Implementation |
|----------|-------------|---------------|
| **Instructor retry** | Single provider, validation failures | `max_retries=3` in Instructor call |
| **Provider fallback** | Need high availability | Try provider A, then B on failure |
| **Schema simplification** | Complex schema keeps failing | Retry with fewer/simpler fields |
| **Chunked extraction** | Long documents | Split input, extract per chunk, merge |

## Exponential Backoff with Jitter

Cap `base_delay * 2 ** attempt` at a `max_delay` and add proportional jitter so concurrent
clients don't retry in lockstep. Before writing this yourself, check the provider SDK --
both the OpenAI and Anthropic clients already retry 429/5xx with backoff internally
(`max_retries`, default 2), so a hand-rolled wrapper around them multiplies attempts:
`max_retries` outer x SDK retries inner.

## Chunked Extraction for Long Documents

```python
from pydantic import BaseModel

def extract_from_chunks(
    document: str,
    schema_cls: type[BaseModel],
    chunk_size: int = 4000,
    overlap: int = 200,
) -> list:
    """Extract structured data from each chunk, return all results."""
    chunks = []
    for i in range(0, len(document), chunk_size - overlap):
        chunks.append(document[i:i + chunk_size])

    results = []
    for chunk in chunks:
        result = extract_single(chunk, schema_cls)
        if result is not None:
            results.append(result)

    return deduplicate(results)
```

## Graceful Degradation

When extraction fails after all retries, degrade gracefully rather than crashing:

```python
from dataclasses import dataclass

@dataclass
class ExtractionResult:
    data: dict | None
    provider: str | None
    is_partial: bool
    error: str | None

def extract_graceful(text: str, schema_cls, fallback_schema_cls=None) -> ExtractionResult:
    """Extract with full fallback chain, always returns a result."""
    # Try full extraction
    result = extract_with_fallback(text, schema_cls)
    if result:
        return ExtractionResult(data=result, provider="auto", is_partial=False, error=None)

    # Try a smaller hand-written schema: required fields only, no nesting.
    # Define it as its own model -- Pydantic has no built-in "simplify" operation.
    if fallback_schema_cls is not None:
        simple_result = extract_with_fallback(text, fallback_schema_cls)
        if simple_result:
            return ExtractionResult(data=simple_result, provider="auto", is_partial=True, error=None)

    # Return empty with error
    return ExtractionResult(data=None, provider=None, is_partial=False, error="All extraction attempts failed")
```
