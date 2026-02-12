---
name: error-handling-patterns
description: Error handling strategy and pattern selection guidance. Use when designing error handling, choosing between exceptions and Result types, or implementing resilience patterns.
---

# Error Handling Patterns

## Pattern Selection Guide

| Pattern | When to Use |
|---------|-------------|
| **Exceptions** | Unexpected failures, I/O errors, truly exceptional conditions |
| **Result/Either types** | Expected failures (validation, parsing), functional codebases |
| **Sentinel errors** | Go; comparison with `errors.Is()` |
| **Error codes** | Cross-boundary APIs, gRPC status codes |
| **Option/Maybe** | Nullable values where absence is normal, not an error |
| **Panic/crash** | Unrecoverable errors, programming bugs, violated invariants |

### Decision Framework
1. Can the caller reasonably recover? -> Result type or checked exception
2. Is this a programming bug? -> Panic/crash (fail fast)
3. Is this crossing a system boundary? -> Error codes with metadata
4. Is this just "no value"? -> Option type, not null

## Error Categories

**Recoverable** (handle gracefully):
- Network timeouts, rate limits -> retry with backoff
- Invalid user input -> validation error with details
- Missing resources -> 404, fallback, or cache
- Transient failures -> circuit breaker

**Unrecoverable** (crash and restart):
- Out of memory, stack overflow
- Corrupted state, violated invariants
- Missing required configuration at startup

## Universal Principles

### 1. Fail Fast
- Validate inputs at system boundaries immediately
- Don't propagate bad data deep into business logic
- Startup: fail if required config/secrets are missing

### 2. Handle at the Right Level
- Catch where you can meaningfully handle (retry, fallback, user message)
- Don't catch just to log and re-throw -- that creates duplicate logs
- Low-level code: propagate errors. High-level code: handle them.

### 3. Preserve Context
- Wrap errors with context: `"failed to create user: <original error>"`
- Include operation, inputs, and timestamp in error metadata
- Use error chaining (`from e` in Python, `%w` in Go, `cause` in Java)

### 4. Error Hierarchy Design
```
ApplicationError (base)
  ├── ValidationError (400)
  ├── NotFoundError (404)
  ├── AuthorizationError (403)
  ├── ConflictError (409)
  └── ExternalServiceError (502)
        ├── service name
        └── original error
```
- Map error types to HTTP status codes at the API boundary
- Include machine-readable `code` field: `"USER_NOT_FOUND"`, `"RATE_LIMITED"`
- Keep user-facing messages separate from developer details

### 5. Don't Swallow Errors
```
# BAD
try:
    do_thing()
except Exception:
    pass  # silent failure

# GOOD
try:
    do_thing()
except SpecificError as e:
    logger.warning(f"Expected failure: {e}")
    return fallback_value
```

### 6. Log Appropriately
- **Error**: Unexpected failures requiring investigation
- **Warning**: Expected failures handled gracefully
- **Don't log**: Every caught exception -- only log when you handle or propagate

## Resilience Patterns

### Retry with Backoff
- Only retry transient errors (network, 503, 429)
- Never retry: 400, 401, 403, 404, 422
- Exponential backoff: `delay * 2^attempt` with jitter
- Max 3 attempts -- more adds latency without improving success rate
- Use `tenacity` (Python), `p-retry` (JS), or language-native constructs

### Circuit Breaker
States: `CLOSED` (normal) -> `OPEN` (failing, reject fast) -> `HALF_OPEN` (testing recovery)

| Parameter | Starting Value |
|-----------|---------------|
| Failure threshold | 5 consecutive failures |
| Open duration | 60 seconds |
| Half-open success threshold | 2 successes to close |

- Apply per external dependency, not globally
- Monitor circuit state transitions as metrics
- Use libraries: `pybreaker`, `opossum` (JS), `gobreaker`

### Graceful Degradation
- Primary -> fallback -> cached value -> default
- Example: live price API -> cached price -> last known price -> "price unavailable"
- Log each fallback step for observability
- Never let a non-critical dependency take down the whole request

### Error Aggregation
- Collect all validation errors before returning (don't fail on first)
- Return all errors at once: `{ errors: [{field: "email", message: "invalid"}, ...] }`
- Use `AggregateError` (JS) or collect into a list

## API Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {"field": "email", "message": "must be a valid email"},
      {"field": "age", "message": "must be >= 18"}
    ],
    "request_id": "req_abc123"
  }
}
```

- Always include `request_id` for debugging
- `code` is machine-readable, `message` is human-readable
- `details` array for multi-field validation errors
- Never expose stack traces or internal paths in production
