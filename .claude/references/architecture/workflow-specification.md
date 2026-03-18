# Workflow Specification

## Workflow Tree Template

Map all execution paths before coding. Every branch must be specified:

```
[Workflow Name]
├── Happy Path
│   ├── Input validated
│   ├── Process executed
│   ├── Side effects committed
│   └── Response returned (200)
├── Validation Failure
│   ├── Input rejected
│   └── Error returned (400) — no side effects
├── Timeout
│   ├── Upstream timeout → retry with backoff OR fail gracefully
│   └── Budget exhausted → return 504 with partial result if safe
├── Transient Failure (retry-eligible)
│   ├── Attempt 1 fails → wait 1s
│   ├── Attempt 2 fails → wait 2s
│   ├── Attempt 3 fails → dead letter queue + alert
│   └── All retries must be idempotent
├── Permanent Failure
│   ├── Unrecoverable error detected
│   ├── Compensation/cleanup triggered
│   └── Error reported (500) + alert
├── Partial Failure (multi-step)
│   ├── Step N succeeds, Step N+1 fails
│   ├── Compensate completed steps (reverse order)
│   └── Report partial state to caller
└── Concurrent Conflict
    ├── Optimistic lock violation detected
    ├── Retry with fresh state OR return 409
    └── Never silently overwrite
```

## Four-View Registry

| View | Question It Answers | Example |
|------|---------------------|---------|
| **By Workflow** | What steps happen, in what order? | "Order placement: validate → charge → reserve → confirm → notify" |
| **By Component** | What workflows touch this component? | "Payment service: order placement, subscription renewal, refund" |
| **By User Journey** | End-to-end from user action to outcome? | "User clicks Buy → sees confirmation → receives email" |
| **By State** | What states can this entity be in? | "Order: draft → submitted → paid → shipped → delivered → returned" |

Build all four views for critical workflows. Cross-reference to catch gaps (e.g., a component touched by a workflow not listed in "by component" view).

## Observable State Discipline

Every state must be visible at three levels:

| Customer View | Operator View | System State |
|---|---|---|
| "Processing your order" | Queue depth: 3, avg wait: 12s | `PAYMENT_PENDING` |
| "Order confirmed" | Payment settled, inventory reserved | `CONFIRMED` |
| "Something went wrong" | Error: payment gateway timeout, retry 2/3 | `PAYMENT_RETRYING` |
| "Refund issued" | Refund processed, inventory released | `REFUNDED` |

**Rule:** If a system state has no customer-facing label, either it's too granular (merge states) or you're hiding information the customer needs.

## Cleanup Inventories

For every resource created during a workflow, document its lifecycle:

| Resource | Created By | Destroyed By | Cleanup Method | Timeout |
|----------|------------|--------------|----------------|---------|
| Payment hold | checkout.initiate | checkout.complete OR checkout.cancel | Release via payment API | 30 min auto-release |
| Reserved inventory | order.confirm | order.ship OR order.cancel | Restore to available pool | 24 hr auto-release |
| Upload temp file | upload.start | upload.process OR upload.fail | Delete from object storage | 1 hr TTL |
| Session token | auth.login | auth.logout OR token.expire | Invalidate in token store | 24 hr expiry |

**Audit rule:** If "Destroyed By" is blank, you have a resource leak.

## Handoff Contracts

When one system hands off to another, specify:

| Field | Value |
|-------|-------|
| **Payload schema** | JSON Schema or type definition link |
| **Error codes** | 4xx/5xx codes caller must handle |
| **Retry policy** | Max attempts, backoff strategy (exponential, jitter) |
| **Timeout budget** | Max time callee has before caller times out |
| **Idempotency key** | Header/field name for safe retries |
| **Dead letter queue** | Where failed messages go after retries exhausted |
| **SLA** | p99 latency, availability target |

## Discovery Audit Checklist

Before specifying a workflow, inventory what exists:

| What to Find | Where to Look | Example Pattern |
|---|---|---|
| API routes | Route files, controllers, OpenAPI spec | `router.post`, `@app.route`, `paths:` |
| Background workers | Job processors, queue consumers | `Bull`, `Celery`, `Sidekiq`, `SQS consumer` |
| Cron jobs | Crontab, scheduler config, periodic tasks | `cron.schedule`, `@periodic_task`, GitHub Actions schedule |
| Event listeners | Pub/sub subscribers, webhook handlers | `on('event')`, `@EventHandler`, SNS subscriptions |
| State machines | Status fields, transition logic | `status enum`, `state_machine`, `transition()` |
| Database triggers | Migration files, DB schema | `CREATE TRIGGER`, `AFTER INSERT` |

## State Machine Template

| State | Valid Transitions | Guard Condition | Side Effect |
|-------|-------------------|-----------------|-------------|
| `DRAFT` | → `SUBMITTED` | All required fields present | Validate, assign ID |
| `SUBMITTED` | → `APPROVED`, → `REJECTED` | Reviewer assigned | Notify reviewer |
| `APPROVED` | → `IN_PROGRESS` | Resources available | Reserve resources |
| `IN_PROGRESS` | → `COMPLETED`, → `FAILED` | — | Execute workflow |
| `COMPLETED` | → `ARCHIVED` | Retention period met | Release resources |
| `FAILED` | → `SUBMITTED` (retry), → `CANCELLED` | — | Compensate, alert |
| `CANCELLED` | (terminal) | — | Release all resources |

**Rules:** Every state must have at least one outbound transition (except terminal states). Every transition must have a guard condition or be unconditional. Side effects must be idempotent if the transition can be retried.
