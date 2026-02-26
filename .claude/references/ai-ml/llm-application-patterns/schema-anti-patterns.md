# Schema Design Anti-Patterns and Validation

Detailed schema anti-patterns and validation strategies for structured LLM output. For the quick-reference tips table, see the [Structured Output section](../SKILL.md#structured-output) in the parent skill.

## Schema Anti-Patterns

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
    revenue: float | None = Field(
        None, description="Annual revenue in USD millions"
    )
    sector: str = Field(
        description="Industry sector",
        json_schema_extra={"enum": ["tech", "finance", "healthcare", "other"]},
    )
    key_products: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Top products/services",
    )
```

### Common Anti-Patterns

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| No field descriptions | Model guesses field semantics | Add `description` to every `Field()` |
| Unconstrained strings | Hallucinated or verbose output | Use `enum`, `max_length`, or regex patterns |
| Too many top-level fields (>15) | Accuracy drops, model loses focus | Group into nested objects by domain |
| Deep nesting (3+ levels) | Models struggle with structure | Flatten or extract in multiple passes |
| Missing optional markers | Model invents values for absent data | Use `T | None` for uncertain fields |
| Generic field names (`f1`, `data`) | Ambiguous extraction targets | Use descriptive domain-specific names |

## Validation Patterns

### Cross-Field Validation

```python
from pydantic import BaseModel, Field, model_validator

class ExtractedEvent(BaseModel):
    event_name: str
    start_date: str = Field(description="ISO 8601 date")
    end_date: str | None = Field(
        None, description="ISO 8601 date, if different from start"
    )
    location: str | None = None
    attendee_count: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def validate_dates(self):
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date cannot be before start_date")
        return self
```

### Field-Level Validation

```python
from pydantic import BaseModel, Field, field_validator

class ContactExtraction(BaseModel):
    email: str = Field(description="Email address")
    phone: str | None = Field(None, description="Phone in E.164 format")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email format")
        return v.lower()

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str | None) -> str | None:
        if v is None:
            return v
        cleaned = "".join(c for c in v if c.isdigit() or c == "+")
        if not cleaned.startswith("+"):
            raise ValueError("Phone must be in E.164 format (+1234567890)")
        return cleaned
```

### Validation with Instructor Retry

When using Instructor, validation errors are automatically fed back to the model on retry. Design validators to produce clear error messages the model can act on:

```python
class InvoiceExtraction(BaseModel):
    line_items: list[LineItem]
    subtotal: float
    tax: float
    total: float

    @model_validator(mode="after")
    def validate_totals(self):
        computed = sum(item.total for item in self.line_items)
        if abs(computed - self.subtotal) > 0.01:
            raise ValueError(
                f"Subtotal {self.subtotal} doesn't match "
                f"sum of line items {computed}"
            )
        if abs(self.subtotal + self.tax - self.total) > 0.01:
            raise ValueError(
                f"Total {self.total} should equal "
                f"subtotal {self.subtotal} + tax {self.tax}"
            )
        return self
```
