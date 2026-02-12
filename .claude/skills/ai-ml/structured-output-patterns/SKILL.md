---
name: structured-output-patterns
description: "Use when extracting structured data from LLMs, implementing JSON mode, function calling, constrained decoding, or building reliable extraction pipelines."
---

# Structured Output Patterns

## Method Selection

| Method | Provider | Guarantees Schema? | Best For |
|--------|----------|-------------------|----------|
| **OpenAI Structured Outputs** | OpenAI | Yes (constrained decoding) | Production extraction with GPT-4o |
| **Anthropic tool_use** | Anthropic | Yes (schema-validated) | Extraction with Claude models |
| **Instructor** | Any (wrapper) | Yes (retry + validation) | Multi-provider, complex validation |
| **Outlines** | Local models | Yes (constrained decoding) | Open-source models, custom grammars |
| **JSON mode** | OpenAI/others | JSON only (no schema) | Simple cases, no strict schema |

**Decision rule**: Use provider-native structured outputs first (OpenAI Structured Outputs or Anthropic tool_use). Use Instructor when you need cross-provider compatibility or complex Pydantic validation. Use Outlines for local/open-source models.

## OpenAI Structured Outputs with Pydantic

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from enum import Enum

class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"

class ReviewAnalysis(BaseModel):
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    key_topics: list[str] = Field(description="Main topics mentioned", max_length=5)
    summary: str = Field(description="One-sentence summary", max_length=200)

client = OpenAI()

completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract structured analysis from the review."},
        {"role": "user", "content": f"Analyze this review: {review_text}"},
    ],
    response_format=ReviewAnalysis,
)

result: ReviewAnalysis = completion.choices[0].message.parsed
print(result.sentiment, result.confidence)
```

### Nested Schemas

```python
class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class ContactInfo(BaseModel):
    email: str | None = None
    phone: str | None = None

class PersonExtraction(BaseModel):
    name: str
    age: int | None = Field(None, description="Age if mentioned")
    addresses: list[Address] = Field(default_factory=list)
    contact: ContactInfo = Field(default_factory=ContactInfo)
    occupation: str | None = None

# Works with nested models -- OpenAI generates valid nested JSON
completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": f"Extract person info: {text}"}],
    response_format=PersonExtraction,
)
```

### Handling Refusals

```python
message = completion.choices[0].message
if message.refusal:
    print(f"Model refused: {message.refusal}")
else:
    result = message.parsed
```

## Anthropic tool_use for Extraction

Force the model to call a "tool" that matches your desired schema. No actual tool execution needed.

```python
import anthropic

client = anthropic.Anthropic()

extraction_tool = {
    "name": "extract_invoice",
    "description": "Extract structured invoice data from the provided text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string", "description": "ISO 8601 date"},
            "vendor_name": {"type": "string"},
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "unit_price": {"type": "number"},
                        "total": {"type": "number"},
                    },
                    "required": ["description", "quantity", "unit_price", "total"],
                },
            },
            "subtotal": {"type": "number"},
            "tax": {"type": "number"},
            "total": {"type": "number"},
            "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
        },
        "required": ["invoice_number", "vendor_name", "line_items", "total", "currency"],
    },
}

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=2048,
    tools=[extraction_tool],
    tool_choice={"type": "tool", "name": "extract_invoice"},  # Force this tool
    messages=[{"role": "user", "content": f"Extract invoice data:\n\n{invoice_text}"}],
)

# Result is in the tool_use block
invoice_data = response.content[0].input  # dict matching the schema
```

### Multiple Extractions

```python
# Extract multiple entities from a single document
extraction_tool = {
    "name": "extract_entities",
    "description": "Extract all people, organizations, and locations mentioned.",
    "input_schema": {
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "mentioned_context": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
            "organizations": {
                "type": "array",
                "items": {"type": "object", "properties": {"name": {"type": "string"}, "type": {"type": "string"}}, "required": ["name"]},
            },
            "locations": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["people", "organizations", "locations"],
    },
}
```

## Instructor Library Patterns

Works with both OpenAI and Anthropic. Adds automatic retry, validation, and streaming.

```bash
pip install instructor
```

### Basic Usage

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

client = instructor.from_openai(OpenAI())

class UserInfo(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)
    email: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower()

user = client.chat.completions.create(
    model="gpt-4o",
    response_model=UserInfo,
    messages=[{"role": "user", "content": f"Extract user info: {text}"}],
)
# user is a validated UserInfo instance
```

### With Anthropic

```python
import instructor
import anthropic

client = instructor.from_anthropic(anthropic.Anthropic())

user = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    response_model=UserInfo,
    messages=[{"role": "user", "content": f"Extract user info: {text}"}],
)
```

### Retry with Validation Feedback

```python
# Instructor automatically retries when validation fails, feeding
# the validation error back to the model (up to max_retries)
user = client.chat.completions.create(
    model="gpt-4o",
    response_model=UserInfo,
    max_retries=3,  # Retries with validation error context
    messages=[{"role": "user", "content": f"Extract: {text}"}],
)
```

### Partial / Streaming Extraction

```python
# Stream partial results as they're generated
for partial_user in client.chat.completions.create_partial(
    model="gpt-4o",
    response_model=UserInfo,
    messages=[{"role": "user", "content": f"Extract: {text}"}],
):
    print(f"Progress: {partial_user}")
    # Fields populate incrementally: UserInfo(name="John", age=None, email=None)
```

### Classification with Enums

```python
from enum import Enum

class TicketCategory(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    feature_request = "feature_request"
    other = "other"

class TicketClassification(BaseModel):
    category: TicketCategory
    priority: int = Field(ge=1, le=5, description="1=lowest, 5=critical")
    requires_human: bool = Field(description="True if this needs human review")
    reasoning: str = Field(description="Brief explanation of classification")

result = client.chat.completions.create(
    model="gpt-4o",
    response_model=TicketClassification,
    messages=[
        {"role": "system", "content": "Classify support tickets accurately."},
        {"role": "user", "content": f"Ticket: {ticket_text}"},
    ],
)
```

## Outlines for Constrained Generation

For local/open-source models. Guarantees schema compliance via constrained decoding (manipulates token logits).

```bash
pip install outlines
```

```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.3")

# JSON schema constraint
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["name", "sentiment", "score"],
}

generator = outlines.generate.json(model, schema)
result = generator(f"Analyze: {text}")
# result is a dict guaranteed to match the schema
```

### Regex Constraints

```python
# Extract dates in exact format
date_generator = outlines.generate.regex(
    model,
    r"\d{4}-\d{2}-\d{2}"
)
date = date_generator("What is today's date? ")
# Output: "2025-01-15" -- guaranteed to match regex
```

### Choice / Classification

```python
classifier = outlines.generate.choice(model, ["positive", "negative", "neutral"])
label = classifier(f"Classify sentiment: {text}")
# Output is guaranteed to be one of the three options
```

## Schema Design Tips

| Tip | Why |
|-----|-----|
| Use `enum` for categorical fields | Prevents hallucinated categories |
| Make uncertain fields `optional` | Model fills None instead of guessing |
| Add `description` to every field | Guides the model on what to extract |
| Use `list[T]` for variable-count entities | Handles 0-N naturally |
| Keep schemas under 15 fields | Accuracy drops with complex schemas |
| Use nested objects for related fields | Groups logically, reduces confusion |

### Schema Anti-Patterns

```python
# BAD: too many top-level fields, no descriptions
class Bad(BaseModel):
    f1: str
    f2: str
    f3: int
    f4: float
    f5: list[str]

# GOOD: descriptive, constrained, grouped
class Good(BaseModel):
    company_name: str = Field(description="Legal company name")
    revenue: float | None = Field(None, description="Annual revenue in USD millions")
    sector: str = Field(description="Industry sector", json_schema_extra={"enum": ["tech", "finance", "healthcare", "other"]})
    key_products: list[str] = Field(default_factory=list, max_length=5, description="Top products/services")
```

## Retry and Fallback Strategies

```python
import time
from pydantic import ValidationError

def extract_with_fallback(text: str, schema_cls, max_retries: int = 3) -> dict | None:
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

## Validation Patterns

```python
from pydantic import BaseModel, Field, model_validator

class ExtractedEvent(BaseModel):
    event_name: str
    start_date: str = Field(description="ISO 8601 date")
    end_date: str | None = Field(None, description="ISO 8601 date, if different from start")
    location: str | None = None
    attendee_count: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def validate_dates(self):
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date cannot be before start_date")
        return self
```

## Gotchas

### Schema Restrictions (OpenAI Structured Outputs)
OpenAI's strict mode requires `additionalProperties: false` on all objects and all fields in `required`. Use Pydantic defaults to handle optional fields -- they'll still be in `required` but the model can output `null`.

### tool_choice Forces a Tool Call
With Anthropic `tool_choice={"type": "tool", "name": "..."}`, the model always calls that tool, even if the input text has nothing to extract. Add validation for empty/garbage extractions.

### Temperature and Structured Output
Use `temperature=0` for extraction tasks. Higher temperature increases the chance of creative but wrong field values. Exception: if you want diverse extractions from ambiguous text.

### Nested Arrays of Objects
Models struggle with deeply nested schemas (3+ levels). Flatten when possible or extract in multiple passes.

### Pydantic V2 vs V1
Instructor and OpenAI SDK require Pydantic V2. If you're on V1, upgrade: `pip install pydantic>=2.0`. Key changes: `@field_validator` replaces `@validator`, `model_dump()` replaces `.dict()`.

### Extraction from Long Documents
For documents exceeding the context window, chunk first and extract from each chunk, then merge/deduplicate results. Don't rely on the model to handle truncation gracefully.

## Cross-References

- **ai-ml:llm-application-patterns** -- prompt engineering, agent tool use, production LLM deployment
- **ai-ml:rag-and-vector-search** -- retrieval pipelines feeding structured extraction
- **languages:pydantic-and-data-validation** -- Pydantic v2 models for extraction schemas
