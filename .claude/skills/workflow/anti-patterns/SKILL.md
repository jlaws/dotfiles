---
name: anti-patterns
description: Code smells and anti-patterns to avoid. Use when reviewing code or debugging issues.
---

# Anti-Patterns

## Code Anti-Patterns

### Premature Abstraction
Wait for 2+ implementations before extracting. Three similar lines is better than a premature abstraction.

```typescript
// Too early: only one use case
class AbstractDataFetcher<T> {
  abstract fetch(): Promise<T>;
  abstract transform(data: T): T;
  abstract validate(data: T): boolean;
}

// Better: concrete implementation first
async function fetchUsers(): Promise<User[]> {
  const data = await api.get('/users');
  return data.filter(u => u.isActive);
}
```

### Over-Engineering
Start simple. Add complexity only when needed.

### God Objects
Split by responsibility. No class should do everything.

### Deep Nesting
More than 2-3 levels indicates need for early returns or extraction.

### Magic Numbers/Strings
Use named constants.

```typescript
// Bad
if (user.role === 'admin') { }
if (retryCount > 3) { }

// Good
const ADMIN_ROLE = 'admin';
const MAX_RETRIES = 3;

if (user.role === ADMIN_ROLE) { }
if (retryCount > MAX_RETRIES) { }
```

## Process Anti-Patterns

### Writing Code Before Understanding
Understand the problem first. Ask clarifying questions.

### Skipping Tests "To Save Time"
Tests save time. Skipping them costs more later.

### Large PRs
Keep PRs small and focused. Hard-to-review PRs get rubber-stamped.

### Catching Exceptions Without Handling
Either handle the error meaningfully or let it propagate.

```typescript
// Bad: swallowing errors
try {
  await saveUser(user);
} catch (e) {
  // silently fails
}

// Good: handle or rethrow
try {
  await saveUser(user);
} catch (e) {
  logger.error('Failed to save user', { error: e, userId: user.id });
  throw new UserSaveError('Failed to save user', { cause: e });
}
```

### Copy-Paste Without Understanding
Understand code before reusing it.

## Communication Anti-Patterns

### Vague Commits
Bad: "fix stuff", "update code", "changes"
Good: "fix: prevent null pointer in user lookup"

### Undocumented Public APIs
Public APIs need documentation. Internal code can be self-documenting.

### TODOs Without Context
Include why, when, and ticket reference.

```typescript
// Bad
// TODO: fix this

// Good
// TODO(#123): handle rate limiting - blocked on API team
```

### Commented-Out Code
Delete it. Git has history.
