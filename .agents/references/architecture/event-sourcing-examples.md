# Event Sourcing Code Examples

## Event Store Schema (PostgreSQL)

```sql
CREATE TABLE events (
    stream_id VARCHAR(255) NOT NULL,
    stream_type VARCHAR(255) NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    event_data JSONB NOT NULL,
    version BIGINT NOT NULL,
    global_position BIGSERIAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_stream_version UNIQUE (stream_id, version)
);

CREATE INDEX idx_events_stream ON events(stream_id, version);
CREATE INDEX idx_events_global ON events(global_position);
CREATE INDEX idx_events_type ON events(event_type);
```

**Key decisions:**
- `stream_id` format: `Order-{uuid}` > `{uuid}` alone (type-based queries)
- `global_position` serial vs timestamp: serial prevents race conditions
- `version` per-stream enables optimistic concurrency

## Projection Rebuild Strategy

```
1. Create new projection table (v2)
2. Replay all events into v2
3. Atomic swap: rename v1 -> old, v2 -> current
4. Drop old after validation
```

Never rebuild in-place -- risks data loss on failure.

## Projection Types

| Type | Use Case | Tradeoff |
|------|----------|----------|
| **Summary view** | Order totals, counts | Must handle out-of-order events |
| **Search index** | Elasticsearch, Algolia | External dependency, harder rebuild |
| **Aggregates** | Daily sales rollups | Time-based bucketing complexity |
| **Denormalized join** | Customer + Orders in one doc | Higher storage, faster queries |

## Saga Compensation Pattern

```python
class SagaOrchestrator:
    """Execute steps with LIFO compensation on failure."""
    def __init__(self):
        self.compensation_stack: list[tuple[str, callable]] = []

    async def execute_step(self, name: str, action: callable, compensate: callable):
        self.compensation_stack.append((name, compensate))
        try:
            return await action()
        except Exception:
            await self.rollback()
            raise

    async def rollback(self):
        while self.compensation_stack:
            name, compensate = self.compensation_stack.pop()
            try:
                await compensate()
            except Exception as e:
                # Compensations must always succeed -- alert and retry
                log.error(f"Compensation failed for {name}: {e}")
                raise
```

## Workflow Engine Constraints (Temporal-style)

**Workflow code (deterministic):**
```python
# PROHIBITED in workflow code
datetime.now()   # Use workflow.now()
random.random()  # Use workflow.random()
# No threading, I/O, or network calls

# Use activities for all side effects
result = await workflow.execute_activity(
    send_email,
    SendEmailInput(to=user.email, template="welcome"),
    start_to_close_timeout=timedelta(seconds=30),
    retry_policy=RetryPolicy(
        maximum_attempts=3,
        non_retryable_error_types=["ValidationError"],
    ),
)
```

**Activity code (non-deterministic):**
- Must be idempotent
- Must have timeout (activities can hang)
- Classify errors: retryable (network) vs non-retryable (validation)
- Use heartbeats for long-running (>30s) activities
- 2MB payload limit per argument

## Do's and Don'ts

### Event Store
- **Do:** Use stream IDs with type prefix (`Order-{uuid}`)
- **Do:** Include correlation/causation IDs in metadata
- **Do:** Implement optimistic concurrency on append
- **Don't:** Update or delete events
- **Don't:** Store large payloads (>10KB)

### CQRS
- **Do:** Denormalize read models for query patterns
- **Do:** Validate in command handlers before state change
- **Do:** Define consistency SLAs per feature
- **Don't:** Query in command handlers
- **Don't:** Couple read/write schemas

### Projections
- **Do:** Make projections idempotent (upsert, not increment)
- **Do:** Store checkpoints for resume
- **Do:** Support full rebuild
- **Don't:** Couple projections (each is independent)
- **Don't:** Ignore projection lag monitoring

### Sagas
- **Do:** Test compensations more than happy path
- **Do:** Use orchestration for >3 steps
- **Do:** Set timeouts on every step
- **Don't:** Skip correlation IDs
- **Don't:** Modify running workflow logic in-place
