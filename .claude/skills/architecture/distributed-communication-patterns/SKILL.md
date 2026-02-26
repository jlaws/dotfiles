---
name: distributed-communication-patterns
description: >
  Design distributed component communication with synchronous RPCs, async messaging, event sourcing, CQRS, and saga orchestration.
  Use when designing how distributed components communicate: synchronous RPCs (gRPC/protobuf),
  asynchronous messaging (Kafka, RabbitMQ, SQS, Celery), event sourcing, CQRS projections,
  or saga/workflow orchestration. Covers technology selection, schema design, reliability
  patterns, and consistency models. Do NOT use for microservice boundary decomposition (use microservices-patterns).
---

# Distributed Communication Patterns

## Top-Level Decision Table

| Need | Pattern | Why |
|------|---------|-----|
| **Request/response, low latency** | gRPC (unary) | Binary, typed, code-gen, 5-10x faster than REST JSON |
| **Streaming data to/from client** | gRPC (streaming) | Native bidirectional; avoids SSE/WS bolt-on |
| **Public API for browsers** | REST or GraphQL | gRPC needs proxy for browsers |
| **Decouple producer from consumer** | Message queue | Async, buffered, retry-friendly |
| **High-throughput event streaming** | Kafka | Millions/s, replay, per-partition ordering |
| **Task distribution with routing** | RabbitMQ | Exchange patterns, <1ms latency, complex routing |
| **Serverless async glue** | SQS | Zero ops, AWS-native, unlimited throughput |
| **Python background tasks** | Celery | Native integration, scheduled jobs, retry built-in |
| **Audit log / temporal queries** | Event sourcing | Immutable event store, full history, replay |
| **Separate read/write models** | CQRS | Optimize reads independently, denormalized views |
| **Multi-step distributed tx** | Saga / workflow engine | Compensation, timeouts, correlation tracking |

## gRPC and Protocol Buffers

### When to Use

- Service-to-service synchronous calls
- Streaming (server, client, or bidirectional)
- Strong schema evolution requirements (field numbers)

### Proto3 Key Conventions

- `syntax = "proto3"` with `package myservice.v1`
- `id` always field 1; `snake_case` field names
- Enums: `ENUM_UNSPECIFIED = 0` always present
- Pagination: `page_size` + opaque `page_token`/`next_page_token`
- Partial updates: `google.protobuf.FieldMask`
- Nullable fields: `google.protobuf.wrappers` (StringValue, Int32Value)

### Service Pattern Types

| Pattern | Proto | Use Case |
|---------|-------|----------|
| Unary | `rpc Get(Req) returns (Resp)` | Standard request/response |
| Server stream | `rpc Watch(Req) returns (stream Event)` | Live feeds, subscriptions |
| Client stream | `rpc BulkCreate(stream Req) returns (Resp)` | Batch uploads |
| Bidirectional | `rpc Chat(stream Msg) returns (stream Msg)` | Real-time collaboration |

### Interceptor Patterns

- **Server auth**: Validate JWT from metadata; skip public methods
- **Client retry**: Retry on UNAVAILABLE/DEADLINE_EXCEEDED with exponential backoff
- **Deadline propagation**: Check `context.time_remaining()`, reserve margin for downstream

### Gotchas

- Field numbers are forever -- use `reserved` for deleted fields
- Default values (0, "", false) not serialized -- can't distinguish unset from default
- Streaming holds connections -- needs gRPC-aware LB (Envoy, Linkerd)
- gRPC-Web needs proxy for browser clients
- Default max message 4MB -- set explicit limits, stream large payloads

> **Code examples**: `references/grpc-examples.md`

## Message Queues

### Technology Selection

| Technology | Throughput | Latency | Ordering | Best For |
|------------|-----------|---------|----------|----------|
| **Kafka** | Millions/s | 5-15ms | Per-partition | Event streaming, log aggregation, replay |
| **RabbitMQ** | 50K/s | <1ms | Per-queue | Task routing, RPC, complex routing rules |
| **Celery** | 10K/s | 5-50ms | None (FIFO opt) | Python task queues, scheduled jobs |
| **SQS** | Unlimited | 20-50ms | FIFO optional | Serverless, AWS-native, zero ops |
| **Redis Streams** | 100K/s | <1ms | Per-stream | Lightweight streaming, ephemeral data |

### Core Reliability Patterns

| Pattern | Implementation |
|---------|---------------|
| **At-least-once** | Default everywhere; design consumers idempotent |
| **Exactly-once (Kafka)** | `enable.idempotence=True` + `acks=all` |
| **Manual commit** | `enable.auto.commit=False`; commit after processing |
| **Publisher confirms** | RabbitMQ `confirm_delivery()` before publish |
| **Late ack** | Celery `task_acks_late=True`; ack after completion |
| **Dead letter queue** | Route failures after N retries; support selective replay |
| **Idempotency key** | Hash of `entity_id:action:timestamp`; Redis-backed with TTL |

### RabbitMQ Exchange Types

| Type | Routing | Use Case |
|------|---------|----------|
| **Fanout** | Broadcast all | Notifications, cache invalidation |
| **Topic** | Pattern match (`order.*`) | Event routing by type |
| **Direct** | Exact key | Targeted task dispatch |

### Gotchas

- Kafka orders only within a partition -- choose partition keys carefully
- Poison messages block queues -- always configure DLQ
- Celery visibility timeout: task redelivered if processing exceeds timeout
- RabbitMQ unbounded queues cause memory pressure -- set `x-max-length`
- SQS FIFO: 300 msg/s per group ID, 3000/s per queue
- Message size limits: SQS 256KB, Kafka 1MB default, RabbitMQ >128KB hurts
- Broker is not a database -- process and persist, don't store long-term

> **Code examples**: `references/message-queue-examples.md`

## Event Sourcing and CQRS

### Event Store Technology Selection

| Technology | Best For | Avoid If |
|------------|----------|----------|
| **EventStoreDB** | Pure event sourcing, built-in projections | Need multi-purpose DB |
| **PostgreSQL** | Existing Postgres stack, SQL expertise | >10K writes/s |
| **Kafka** | High-throughput streaming + event bus | Per-aggregate queries critical |
| **DynamoDB** | Serverless, AWS-native, auto-scaling | Complex cross-stream queries |

### Event Store Guardrails

- **Immutability**: Never UPDATE/DELETE events -- add compensating events
- **Optimistic concurrency**: Check `expected_version` on append
- **Event size**: Keep <10KB; reference large payloads via URL/S3 key
- **Idempotency**: Use `event_id` for deduplication
- **Schema versioning**: Add `schema_version` from day one; upcasting is harder later
- **Stream ID format**: `{Type}-{UUID}` enables type-based queries
- **Correlation/causation IDs**: Required for tracing

### CQRS Consistency Models

| Model | When | Implementation |
|-------|------|----------------|
| **Eventual** | Default | Async projections, no write-time coupling |
| **Read-your-writes** | User expects immediate visibility | Poll projection until version >= write version (5s timeout) |
| **Inline projection** | Strong consistency required | Update read model in same transaction (couples stores) |

### Projection Patterns

- **Idempotent**: Upsert with full state, not incremental updates
- **Checkpointed**: Store `last_processed_global_position` per projection
- **Rebuildable**: New table -> replay all events -> atomic swap -> drop old
- **Types**: Summary view, search index, aggregates, denormalized joins

### Saga and Workflow Orchestration

| Factor | Choreography | Orchestration |
|--------|--------------|---------------|
| **Coupling** | Loose (react to events) | Tighter (orchestrator knows steps) |
| **Visibility** | Hard to trace | Orchestrator holds state |
| **Complexity ceiling** | Breaks at 4+ steps | Scales to 10+ steps |

**Default to orchestration** unless <4 steps with simple compensation.

**Compensation design (LIFO)**:
- Register compensation before each step execution
- Compensate in reverse order (stack, not queue)
- Compensations must be idempotent and always succeed
- Track completed steps; only compensate those

### Workflow Engine Constraints (Temporal-style)

- **Workflow code**: Deterministic only -- no `datetime.now()`, `random()`, I/O
- **Activity code**: Idempotent, always has timeout, heartbeat if >30s
- **Error classification**: Retryable (network) vs non-retryable (validation)
- **Versioning**: Never modify running workflow logic; use version gates

### Gotchas

- Eventual consistency needs SLAs -- define acceptable lag and monitor breaches
- Don't query in command handlers -- breaks CQRS separation
- Projection lag can snowball -- needs backpressure or scaling
- Choreography still needs saga ID for correlation across events
- Business logic belongs in workflows, not activities (I/O adapters)

> **Code examples**: `references/event-sourcing-examples.md`

## Event Schema Design

Standard envelope for all domains:

```python
@dataclass
class DomainEvent:
    event_type: str
    aggregate_id: str
    data: dict
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    schema_version: int = 1
    correlation_id: str = ""
    idempotency_key: str = ""
```

## Idempotency Key Pattern

```python
def generate_idempotency_key(entity_id: str, action: str, timestamp: str) -> str:
    return hashlib.sha256(f"{entity_id}:{action}:{timestamp}".encode()).hexdigest()
```

Redis-backed check: `SET idempotency:{key} 1 NX EX 86400` -- returns None if duplicate.
