# Provider Code Examples

Extended code examples for each structured output provider. For method selection and quick-reference, see the [Structured Output section](../llm-application-patterns.md#structured-output) in the parent reference.

`OPENAI_MODEL` and `MODEL` (Anthropic) below stand in for whichever model IDs you configure. Read model IDs from
config rather than scattering literals: an ID names one model and does not advance to the
next generation, so the upgrade is a one-line config change.

## OpenAI Structured Outputs

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
    model=OPENAI_MODEL,
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

## Anthropic Structured Outputs -- Multiple Extractions

Extract multiple entity types from a single document with one nested model.

```python
import anthropic
from pydantic import BaseModel

MODEL = "claude-sonnet-5-5"  # from config

class Person(BaseModel):
    name: str
    role: str | None = None
    mentioned_context: str | None = None

class Organization(BaseModel):
    name: str
    type: str | None = None

class Entities(BaseModel):
    people: list[Person]
    organizations: list[Organization]
    locations: list[str]

client = anthropic.Anthropic()
response = client.messages.parse(
    model=MODEL,
    max_tokens=16000,
    messages=[{"role": "user", "content": f"Extract all people, organizations, and locations mentioned:\n\n{text}"}],
    output_format=Entities,
)
entities = response.parsed_output  # Entities, or None on refusal or truncation
if entities is None:
    raise ValueError(f"no structured output (stop_reason={response.stop_reason})")
```

## Instructor Library Patterns

Works with both OpenAI and Anthropic. Adds automatic retry, validation, and streaming.

```bash
pip install "instructor>=1.17"
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
    model=OPENAI_MODEL,
    response_model=UserInfo,
    messages=[{"role": "user", "content": f"Extract user info: {text}"}],
)
# user is a validated UserInfo instance
```

### With Anthropic

```python
import instructor
import anthropic

# JSON_SCHEMA uses Anthropic structured outputs. The default tools mode forces a tool
# call, which current Claude models reject.
client = instructor.from_anthropic(anthropic.Anthropic(), mode=instructor.Mode.JSON_SCHEMA)

user = client.messages.create(
    model=MODEL,
    max_tokens=16000,
    response_model=UserInfo,
    messages=[{"role": "user", "content": f"Extract user info: {text}"}],
)
```

### Retry with Validation Feedback

```python
# Instructor automatically retries when validation fails, feeding
# the validation error back to the model (up to max_retries)
user = client.chat.completions.create(
    model=OPENAI_MODEL,
    response_model=UserInfo,
    max_retries=3,  # Retries with validation error context
    messages=[{"role": "user", "content": f"Extract: {text}"}],
)
```

### Partial / Streaming Extraction

```python
# Stream partial results as they're generated
for partial_user in client.chat.completions.create_partial(
    model=OPENAI_MODEL,
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
    model=OPENAI_MODEL,
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

### JSON Schema Constraint

```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.3")

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
