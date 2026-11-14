# Architecture Patterns

## Clean Architecture

### Directory Structure
```
app/
├── domain/           # Entities & business rules
│   ├── entities/
│   ├── value_objects/
│   └── interfaces/   # Abstract interfaces (ports)
├── use_cases/        # Application business rules
├── adapters/         # Interface implementations
│   ├── repositories/
│   ├── controllers/
│   └── gateways/
└── infrastructure/   # Framework & external concerns
```

The same dependency rule underlies Hexagonal Architecture (ports as interfaces, adapters at the
edges) and Domain-Driven Design (value objects, entities with identity, aggregate roots reached
through repositories). All three keep frameworks out of the core; they differ mainly in what they
name things.

## Architecture Selection Guide

### Reversibility-First Principle

Prefer decisions that are easy to change over ones that are "optimal." Architecture is evolutionary, not permanent. When choosing between two approaches of similar merit, pick the one that's cheaper to reverse.

**Irreversibility spectrum** (least → most): Database schema < API contract < data model < service boundary < programming language

### When NOT to Use a Pattern

| Pattern | Avoid When |
|---|---|
| **Microservices** | Small team (<5 devs), early-stage product, <3 bounded contexts, no independent deployment need |
| **Event Sourcing** | Simple CRUD, no audit requirements, team unfamiliar with eventual consistency |
| **Clean Architecture** | Simple scripts/tools, prototype/throwaway code, <3 entities |
| **CQRS** | Read and write models are identical, low traffic, no complex reporting needs |
| **DDD** | Simple domain (no business rules beyond validation), solo developer, throwaway prototype |
| **Hexagonal Architecture** | Single adapter per port (over-abstraction), no planned adapter swaps |

## Key Principles

1. **Dependency Rule**: Dependencies always point inward
2. **Interface Segregation**: Small, focused interfaces
3. **Business Logic in Domain**: Keep frameworks out of core
4. **Test Independence**: Core testable without infrastructure
5. **Rich Domain Models**: Behavior with data, not anemic entities

## Module Depth

- **The interface is the test surface**: test at the deepest interface, then delete the shallow unit tests it replaces.
- **Two adapters make a seam**: one adapter is a hypothetical seam. Do not add a port for one implementation; the test double counts as the second adapter.
- **Classify each dependency**: in-process (call it), local and substitutable (use the real thing in tests), remote and owned (port plus adapter), true external (mock at the boundary).

Source: John Ousterhout, *A Philosophy of Software Design*, via mattpocock/skills `codebase-design`.

## Pitfalls

- **Anemic Domain**: Entities with only data, no behavior
- **Framework Coupling**: Business logic depends on frameworks
- **Fat Controllers**: Business logic in controllers
- **Repository Leakage**: Exposing ORM objects
- **Over-Engineering**: Clean architecture for simple CRUD
