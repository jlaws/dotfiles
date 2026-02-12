---
name: event-sourcing-patterns
description: Use when designing event stores, implementing CQRS read/write separation, building projections, or coordinating distributed sagas. Covers technology selection, consistency handling, and workflow orchestration patterns.
---

# Event Sourcing & CQRS Patterns

## Event Store Technology Selection

| Technology | Best For | Avoid If |
|------------|----------|----------|
| **EventStoreDB** | Pure event sourcing, projections built-in | Need multi-purpose DB |
| **PostgreSQL** | Existing Postgres stack, SQL expertise | Extreme write throughput (>10K/s) |
| **Kafka** | High-throughput streaming, event bus | Per-aggregate queries critical |
| **DynamoDB** | Serverless, AWS-native, auto-scaling | Complex cross-stream queries |

## Event Store Schema Design (Postgres)

```sql
CREATE TABLE events (
    stream_id VARCHAR(255) NOT NULL,  -- Pattern: "{Type}-{UUID}"
    stream_type VARCHAR(255) NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    event_data JSONB NOT NULL,
    version BIGINT NOT NULL,
    global_position BIGSERIAL,        -- Critical for projections
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_stream_version UNIQUE (stream_id, version)
);

CREATE INDEX idx_events_stream ON events(stream_id, version);
CREATE INDEX idx_events_global ON events(global_position);  -- Projection catchup
CREATE INDEX idx_events_type ON events(event_type);         -- Type-based subscriptions
```

**Key Decisions:**
- `stream_id` format: `Order-{uuid}` > `{uuid}` alone (enables type-based queries)
- `global_position` serial vs timestamp: serial prevents race conditions
- `version` per-stream vs global: per-stream enables optimistic concurrency

## Event Store Guardrails

- **Immutability**: Never UPDATE or DELETE events -- add compensating events instead
- **Optimistic concurrency**: Always check `expected_version` on append to prevent lost updates
- **Event size**: Keep <10KB; reference large payloads via URL/S3 key
- **Idempotency**: Use `event_id` for deduplication; append must be idempotent
- **Correlation/Causation IDs**: Required for tracing -- `metadata.correlation_id`

## CQRS: Consistency Models

| Model | When | Implementation |
|-------|------|----------------|
| **Eventual** | Default -- acceptable lag | Async projections, no write-time coupling |
| **Read-your-writes** | User expects immediate visibility | Poll projection until version ≥ write version (5s timeout) |
| **Inline projection** | Strong consistency required | Update read model in same transaction as event append |

**Gotcha:** Inline projections couple write/read stores -- breaks scaling independence. Only for single-DB deployments.

## Projection Design Patterns

### Idempotency

Projections **must** be idempotent (events replay during catchup/rebuild). Techniques:
- Upsert with full state (not incremental updates)
- Track `last_processed_version` per entity
- Use `ON CONFLICT DO UPDATE` (Postgres) or conditional expressions (DynamoDB)

### Checkpointing

Store `last_processed_global_position` per projection:
- Enables resume after restart
- Supports independent projection versioning
- **Checkpoint frequency:** every event is overkill; batch every 100-1000 events

### Rebuild Strategy

Projections must support full rebuild:
1. Create new projection table (v2)
2. Replay all events into v2
3. Atomic swap: rename v1→old, v2→current
4. Drop old after validation

**Never** rebuild in-place -- risks data loss on failure.

### Projection Types

| Type | Use Case | Tradeoff |
|------|----------|----------|
| **Summary view** | Order totals, counts | Must handle out-of-order events |
| **Search index** | Elasticsearch, Algolia | External dependency, harder rebuild |
| **Aggregates** | Daily sales rollups | Time-based bucketing complexity |
| **Denormalized join** | Customer + Orders in one doc | Higher storage, faster queries |

## Saga & Workflow Orchestration

### Choreography vs Orchestration

| Factor | Choreography | Orchestration |
|--------|--------------|---------------|
| **Coupling** | Loose (services react to events) | Tighter (orchestrator knows steps) |
| **Visibility** | Hard to trace end-to-end | Easy (orchestrator holds state) |
| **Complexity ceiling** | Breaks down at 4+ steps | Scales to 10+ steps |
| **Best for** | 2-3 steps, decoupled teams | Order-dependent steps, complex compensation |

**Default to orchestration** unless <4 steps with simple compensation.

### Saga vs Workflow Engine

| Feature | Plain Saga | Workflow Engine (Temporal) |
|---------|-----------|----------------------------|
| **Retries** | Manual | Built-in with backoff |
| **State persistence** | Manual saga store | Automatic |
| **Determinism** | Not enforced | Enforced (replay-safe) |
| **Versioning** | Manual migration | `workflow.get_version()` |
| **Pick when** | Simple compensating txns | Long-running, stateful, multi-service |

### Compensation Design (Critical)

- **LIFO order**: Compensate in reverse execution order (stack, not queue)
- **Idempotency**: Compensations retry; design for multiple execution
- **Always succeed**: Compensations cannot fail -- if they can, add retry/alert
- **Register before execution**: Add compensation to stack before each step
- **Partial compensation**: Track completed steps; only compensate those

**Gotcha:** A saga that completes compensation is "failed successfully" -- distinguish from unrecoverable failures in monitoring.

### Workflow Engine Constraints (Temporal-style)

**Workflow Code (Deterministic):**
- Prohibited: `datetime.now()`, `random()`, threading, I/O, network
- Use instead: `workflow.now()`, `workflow.random()`, activities for side effects

**Activity Code (Non-Deterministic):**
- Must be idempotent
- Must have timeout (activities can hang)
- Classify errors: retryable (network) vs non-retryable (validation)
- Use heartbeats for long-running (>30s) activities
- 2MB payload limit per argument

**Gotcha:** Workflow code runs repeatedly during replay -- any non-determinism causes divergence errors.

### Operational Guardrails

- **Correlation IDs**: Propagate through every step for tracing
- **Timeouts on every step**: Never wait indefinitely (5min default)
- **Monitor**: workflow duration, step failure rate, compensation trigger rate, stuck count
- **Versioning**: Never modify running workflow logic; use version gates or new workflow types

## Non-Obvious Gotchas

- **Choreography needs saga ID:** Even without orchestrator, need correlation across events
- **Eventual consistency SLAs:** Define acceptable lag (500ms? 5s?) and monitor breach rate
- **Event versioning from day one:** Add `event.schema_version`; upcasting is harder than prevention
- **Don't query in command handlers:** Commands are for writes; breaks CQRS separation
- **Business logic in workflows, not activities:** Activities are I/O adapters; decisions belong in workflow
- **Projection lag snowball:** If projection falls behind, writes accelerate lag -- needs backpressure or scaling

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
- **Do:** Make projections idempotent
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
