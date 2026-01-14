---
name: clean-code
description: Clean code principles and quality standards. Use when writing, reviewing, or refactoring code.
---

# Clean Code Principles

## Core Principles

### Single Responsibility
One reason to change per function/class.

```typescript
// Bad: handles validation AND persistence
function saveUser(data: unknown) {
  if (!data.email) throw new Error('Email required');
  if (!data.name) throw new Error('Name required');
  return db.users.insert(data);
}

// Good: separate concerns
function validateUser(data: unknown): User {
  if (!data.email) throw new Error('Email required');
  if (!data.name) throw new Error('Name required');
  return data as User;
}

function saveUser(user: User) {
  return db.users.insert(user);
}
```

### DRY (Don't Repeat Yourself)
Extract duplicates, but don't over-abstract prematurely. Wait for 2+ implementations before extracting.

### YAGNI (You Aren't Gonna Need It)
Don't build until needed. Solve today's problem, not tomorrow's hypothetical.

### Composition over Inheritance
Prefer protocols/interfaces and composition.

```swift
// Prefer
protocol Persistable { func save() async throws }
struct UserService: Persistable { }

// Over
class BaseService { func save() { } }
class UserService: BaseService { }
```

### Explicit over Implicit
Clarity beats cleverness. Readable code > clever one-liners.

## Quality Standards

| Rule | Language |
|------|----------|
| No `any` type | TypeScript (use `unknown`) |
| No force unwraps | Swift (unless provably safe) |
| Type hints required | Python (public functions) |
| Explicit error handling | Go, Rust |
| All public APIs documented | All |

## Refactoring Triggers

| Smell | Action |
|-------|--------|
| Function > 30 lines | Extract smaller functions |
| > 3 parameters | Use parameter object |
| Nested conditionals > 2 levels | Use early returns |
| Duplicated code > 2x | Extract utility |

### Early Returns

```typescript
// Bad: nested conditionals
function process(user: User | null) {
  if (user) {
    if (user.isActive) {
      if (user.hasPermission) {
        return doWork(user);
      }
    }
  }
  return null;
}

// Good: early returns
function process(user: User | null) {
  if (!user) return null;
  if (!user.isActive) return null;
  if (!user.hasPermission) return null;
  return doWork(user);
}
```

### Parameter Objects

```typescript
// Bad: too many params
function createUser(
  name: string,
  email: string,
  age: number,
  role: string,
  department: string
) { }

// Good: parameter object
interface CreateUserParams {
  name: string;
  email: string;
  age: number;
  role: string;
  department: string;
}

function createUser(params: CreateUserParams) { }
```
