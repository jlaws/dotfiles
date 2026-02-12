---
name: openapi-spec-generation
description: Generate and maintain OpenAPI 3.1 specifications from code, design-first specs, and validation patterns. Use when creating API documentation, generating SDKs, or ensuring API contract compliance.
---

# OpenAPI Spec Generation

## Design Approaches

| Approach | Description | Best For |
|----------|-------------|----------|
| **Design-First** | Write spec before code | New APIs, contracts |
| **Code-First** | Generate spec from code | Existing APIs |
| **Hybrid** | Annotate code, generate spec | Evolving APIs |

## Template: Complete API Specification

```yaml
openapi: 3.1.0
info:
  title: User Management API
  version: 2.0.0
  contact: { name: API Support, email: api-support@example.com }

servers:
  - url: https://api.example.com/v2
  - url: http://localhost:3000/v2

tags:
  - { name: Users, description: User management }

paths:
  /users:
    get:
      operationId: listUsers
      tags: [Users]
      parameters:
        - $ref: '#/components/parameters/PageParam'
        - $ref: '#/components/parameters/LimitParam'
        - { name: status, in: query, schema: { $ref: '#/components/schemas/UserStatus' } }
        - { name: search, in: query, schema: { type: string, minLength: 2, maxLength: 100 } }
      responses:
        '200': { content: { application/json: { schema: { $ref: '#/components/schemas/UserListResponse' } } } }
        '400': { $ref: '#/components/responses/BadRequest' }
        '401': { $ref: '#/components/responses/Unauthorized' }
      security: [{ bearerAuth: [] }]

    post:
      operationId: createUser
      tags: [Users]
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: '#/components/schemas/CreateUserRequest' }
      responses:
        '201': { content: { application/json: { schema: { $ref: '#/components/schemas/User' } } },
                 headers: { Location: { schema: { type: string, format: uri } } } }
        '409': { content: { application/json: { schema: { $ref: '#/components/schemas/Error' } } } }
      security: [{ bearerAuth: [] }]

  /users/{userId}:
    parameters: [{ $ref: '#/components/parameters/UserIdParam' }]
    get:
      operationId: getUser
      tags: [Users]
      responses:
        '200': { content: { application/json: { schema: { $ref: '#/components/schemas/User' } } } }
        '404': { $ref: '#/components/responses/NotFound' }
    patch:
      operationId: updateUser
      tags: [Users]
      requestBody: { required: true, content: { application/json: { schema: { $ref: '#/components/schemas/UpdateUserRequest' } } } }
      responses:
        '200': { content: { application/json: { schema: { $ref: '#/components/schemas/User' } } } }
    delete:
      operationId: deleteUser
      tags: [Users]
      responses: { '204': { description: User deleted } }

components:
  schemas:
    User:
      type: object
      required: [id, email, name, status, createdAt]
      properties:
        id: { type: string, format: uuid, readOnly: true }
        email: { type: string, format: email }
        name: { type: string, minLength: 1, maxLength: 100 }
        status: { $ref: '#/components/schemas/UserStatus' }
        role: { type: string, enum: [user, moderator, admin], default: user }
        avatar: { type: string, format: uri, nullable: true }
        metadata: { type: object, additionalProperties: true }
        createdAt: { type: string, format: date-time, readOnly: true }

    UserStatus: { type: string, enum: [active, inactive, suspended, pending] }

    CreateUserRequest:
      type: object
      required: [email, name]
      properties:
        email: { type: string, format: email }
        name: { type: string, minLength: 1, maxLength: 100 }
        role: { type: string, enum: [user, moderator, admin], default: user }

    UpdateUserRequest:
      type: object
      minProperties: 1
      properties:
        name: { type: string, minLength: 1, maxLength: 100 }
        status: { $ref: '#/components/schemas/UserStatus' }
        role: { type: string, enum: [user, moderator, admin] }

    UserListResponse:
      type: object
      required: [data, pagination]
      properties:
        data: { type: array, items: { $ref: '#/components/schemas/User' } }
        pagination: { $ref: '#/components/schemas/Pagination' }

    Pagination:
      type: object
      required: [page, limit, total, totalPages]
      properties:
        page: { type: integer, minimum: 1 }
        limit: { type: integer, minimum: 1, maximum: 100 }
        total: { type: integer, minimum: 0 }
        totalPages: { type: integer, minimum: 0 }
        hasNext: { type: boolean }
        hasPrev: { type: boolean }

    Error:
      type: object
      required: [code, message]
      properties:
        code: { type: string }
        message: { type: string }
        details: { type: array, items: { type: object, properties: { field: { type: string }, message: { type: string } } } }
        requestId: { type: string }

  parameters:
    UserIdParam: { name: userId, in: path, required: true, schema: { type: string, format: uuid } }
    PageParam: { name: page, in: query, schema: { type: integer, minimum: 1, default: 1 } }
    LimitParam: { name: limit, in: query, schema: { type: integer, minimum: 1, maximum: 100, default: 20 } }

  responses:
    BadRequest: { description: Invalid request, content: { application/json: { schema: { $ref: '#/components/schemas/Error' } } } }
    Unauthorized: { description: Authentication required, content: { application/json: { schema: { $ref: '#/components/schemas/Error' } } } }
    NotFound: { description: Resource not found, content: { application/json: { schema: { $ref: '#/components/schemas/Error' } } } }
    RateLimited:
      description: Too many requests
      headers:
        Retry-After: { schema: { type: integer } }
        X-RateLimit-Limit: { schema: { type: integer } }
        X-RateLimit-Remaining: { schema: { type: integer } }

  securitySchemes:
    bearerAuth: { type: http, scheme: bearer, bearerFormat: JWT }
    apiKey: { type: apiKey, in: header, name: X-API-Key }

security: [{ bearerAuth: [] }]
```

## Code-First: Python/FastAPI

```python
from fastapi import FastAPI, Query, Path
from pydantic import BaseModel, Field, EmailStr

app = FastAPI(title="User Management API", version="2.0.0")

class UserCreate(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)
    role: str = Field(default="user")

class User(BaseModel):
    id: UUID
    email: EmailStr
    name: str
    status: str
    created_at: datetime = Field(..., alias="createdAt")

@app.get("/users", response_model=UserListResponse, tags=["Users"])
async def list_users(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
): pass

@app.post("/users", response_model=User, status_code=201, tags=["Users"])
async def create_user(user: UserCreate): pass

# Export: python -c "import json; from main import app; print(json.dumps(app.openapi(), indent=2))"
```

## Validation & Linting

```yaml
# .spectral.yaml
extends: ["spectral:oas"]
rules:
  operation-operationId: error
  operation-description: warn
  operation-security-defined: error
  path-params-snake-case:
    severity: warn
    given: "$.paths[*].parameters[?(@.in == 'path')].name"
    then: { function: pattern, functionOptions: { match: "^[a-z][a-z0-9_]*$" } }
```

```bash
# Spectral
npm install -g @stoplight/spectral-cli
spectral lint openapi.yaml

# Redocly
npm install -g @redocly/cli
redocly lint openapi.yaml
redocly bundle openapi.yaml -o bundled.yaml
redocly preview-docs openapi.yaml
```

## SDK Generation

```bash
npm install -g @openapitools/openapi-generator-cli

# TypeScript client
openapi-generator-cli generate -i openapi.yaml -g typescript-fetch -o ./generated/ts-client

# Python client
openapi-generator-cli generate -i openapi.yaml -g python -o ./generated/py-client

# Go client
openapi-generator-cli generate -i openapi.yaml -g go -o ./generated/go-client
```
