---
name: pydantic-and-data-validation
description: Build robust data models with Pydantic v2 validators, serialization, and settings management
---

# Pydantic and Data Validation

Pydantic v2 model patterns, custom validators, discriminated unions, settings management, serialization, and FastAPI/SQLAlchemy integration.

## Validation Library Decision Table

| Use Case | Library | Why |
|----------|---------|-----|
| API request/response models | **Pydantic** | FastAPI native, JSON Schema, rich validation |
| Internal data containers, no validation | `dataclasses` | Stdlib, zero deps, fastest construction |
| High-throughput deserialization | `msgspec` | 5-10x faster than Pydantic, Struct-based |
| Attrs ecosystem / legacy | `attrs` | Mature, flexible, slots by default |
| Config from env vars / files | **Pydantic Settings** | Layered sources, type coercion, .env support |
| Schema-first (OpenAPI import) | `datamodel-code-generator` | Generates Pydantic models from JSON Schema |
| DataFrame validation | `pandera` | Column-level checks on pandas/polars |

## Core Model Patterns

### Field and Model Validators

```python
from pydantic import BaseModel, field_validator, model_validator, Field, ConfigDict
from datetime import date

class User(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str
    age: int = Field(ge=0, le=150)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        # Runs after default validation; v is already a str
        if "@" not in v:
            raise ValueError("must contain @")
        return v.lower().strip()

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, v: str) -> str:
        # mode="before" runs before type coercion
        return v.strip() if isinstance(v, str) else v

class DateRange(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def check_range(self) -> "DateRange":
        if self.start >= self.end:
            raise ValueError("start must precede end")
        return self
```

### Computed Fields and Serialization

```python
from pydantic import computed_field, ConfigDict

class Product(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,     # accept field name OR alias
        str_strip_whitespace=True, # strip all str fields
        frozen=True,               # immutable after creation
    )
    price_cents: int
    tax_rate: float = 0.08

    @computed_field
    @property
    def total_cents(self) -> int:
        return int(self.price_cents * (1 + self.tax_rate))

# Serialization control
p = Product(price_cents=1000)
p.model_dump(exclude={"tax_rate"})  # {'price_cents': 1000, 'total_cents': 1080}
p.model_dump(mode="json")           # JSON-safe types (datetimes -> str)
Product.model_json_schema()          # Full JSON Schema dict
```

## Custom Types with Annotated

```python
from typing import Annotated
from pydantic import AfterValidator, BeforeValidator, PlainSerializer

def _check_positive(v: int) -> int:
    if v <= 0:
        raise ValueError("must be positive")
    return v

PositiveInt = Annotated[int, AfterValidator(_check_positive)]

# Coercion: comma-separated string -> list
def _split_tags(v):
    if isinstance(v, str):
        return [t.strip() for t in v.split(",") if t.strip()]
    return v

TagList = Annotated[list[str], BeforeValidator(_split_tags)]
SecretStr = Annotated[str, PlainSerializer(lambda v: "***", return_type=str)]

class Config(BaseModel):
    retries: PositiveInt
    tags: TagList
    api_key: SecretStr
```

## Discriminated Unions

```python
from typing import Literal, Union, Annotated

class CreditCard(BaseModel):
    method: Literal["credit_card"]
    card_number: str
    exp_month: int

class BankTransfer(BaseModel):
    method: Literal["bank_transfer"]
    routing_number: str
    account_number: str

# Pydantic checks "method" field to pick correct submodel
PaymentMethod = Annotated[
    Union[CreditCard, BankTransfer],
    Field(discriminator="method"),
]

class Order(BaseModel):
    total: float
    payment: PaymentMethod  # automatic dispatch by "method" value
```

## BaseSettings with Env Vars

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",           # APP_DB_HOST -> db_host
        env_nested_delimiter="__",   # APP_DB__HOST -> db.host
        case_sensitive=False,
    )
    db_host: str = "localhost"
    db_port: int = 5432
    db_password: SecretStr           # never printed in repr
    debug: bool = False
    allowed_origins: list[str] = ["http://localhost:3000"]

# Priority: env vars > .env file > defaults
settings = AppSettings()
# Access secret: settings.db_password.get_secret_value()
```

## FastAPI Integration

```python
from fastapi import FastAPI, Depends, Query

class ItemCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    price: float = Field(gt=0)
    tags: list[str] = []

class ItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # read from ORM objects
    id: int
    name: str
    price: float

app = FastAPI()

@app.post("/items", response_model=ItemResponse, status_code=201)
async def create_item(
    item: ItemCreate,                              # body, auto-validated
    settings: AppSettings = Depends(AppSettings),  # inject config
):
    db_item = await save_to_db(item.model_dump())
    return db_item  # auto-serialized via from_attributes
```

## SQLAlchemy Integration

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class UserORM(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    email: Mapped[str]

class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: str
# Usage: UserRead.model_validate(user_orm_instance)

class UserWrite(BaseModel):
    name: str = Field(min_length=1)
    email: str
    def to_orm(self) -> UserORM:
        return UserORM(**self.model_dump())
```

### Alias Generators for API Style

```python
from pydantic.alias_generators import to_camel

class CamelModel(BaseModel):
    """Base model that accepts and emits camelCase JSON."""
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

class UserProfile(CamelModel):
    first_name: str
    last_name: str
    is_active: bool = True
# Accepts: {"firstName": "Alice", "lastName": "Smith"}
# model_dump(by_alias=True) -> {"firstName": "Alice", ...}
```

## Gotchas

- **v1 vs v2**: `validator` -> `field_validator`, `root_validator` -> `model_validator`, `Config` class -> `model_config = ConfigDict(...)`. v2 is a full rewrite; avoid the v1 compat layer in new code.
- **`mode="before"` receives raw input**: could be `dict`, `str`, `None` -- always type-check before operating on the value.
- **`ConfigDict(strict=True)`**: disables coercion (`"123"` won't become `int`). Use per-field `Strict` types for granular control.
- **Mutable defaults**: `list[str] = []` is safe in Pydantic (deep-copied per instance), unlike plain Python classes.
- **`from_attributes=True`** must be set on the Pydantic model, not the ORM model. Without it, `model_validate(orm_obj)` raises.
- **SecretStr**: `.get_secret_value()` to read; `repr()` and `model_dump()` hide it. `model_dump(mode="json")` still redacts.
- **Discriminated unions require `Literal` type** on the discriminator field in every variant. Missing it produces cryptic validation errors.
- **JSON Schema**: `model_json_schema()` excludes computed fields by default. Pass `mode="serialization"` to include them.
- **Circular references**: use `model_rebuild()` after all models are defined, or `from __future__ import annotations`.
