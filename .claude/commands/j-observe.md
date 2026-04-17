---
name: j-observe
description: "Observability consultation -- logging, metrics, tracing, alerting, and analytics instrumentation. Use when designing observability systems, adding instrumentation, setting up dashboards, or defining SLOs. Do NOT use for CI/CD pipeline config (use /j-devops) or cloud infrastructure (use /j-cloud)."
argument-hint: "<question-or-task>"
---

Load skill `analysis-output-patterns` for output structure rules.

Before starting, gather diagnostic context:

1. **Detect observability stack** from config files and dependencies (OpenTelemetry, Prometheus, Grafana, Datadog, New Relic, Sentry, ELK/OpenSearch, Loki, Tempo, Jaeger).
2. **Identify logging setup** by searching for logger configuration, structured logging libraries (winston, pino, structlog, slog, zerolog, log4j, serilog), or log format definitions.
3. **Check metrics/tracing instrumentation** by searching for metric registries, histogram/counter definitions, span creation, or OTel SDK initialization.
4. **Detect analytics tracking** by searching for event tracking calls, analytics providers (Segment, Mixpanel, Amplitude, PostHog), or tracking plan definitions.
5. **Get scope overview** of the target area (if $ARGUMENTS specifies a service or component, scope to that; otherwise scan for monitoring/, observability/, telemetry/, instrumentation/, or analytics/ directories and config).

Load relevant references based on what the diagnostic context reveals:
- `references/devops/observability` -- golden signals, metric design, tracing strategy, alerting, stack preferences
- `references/devops/sre-practices` -- error budgets, SLO targets by tier, progressive deployment
- `references/devops/incident-management` -- severity framework, runbook structure, postmortem process
- `references/business/analytics-instrumentation` -- event taxonomy, tracking wrappers, funnel design, PII gotchas
- `references/architecture/error-handling-patterns` -- error hierarchy, logging levels, resilience patterns
- `references/business/kpi-dashboard-design` -- dashboard hierarchy, KPI framework

Help with: $ARGUMENTS

---

## Structured Logging Best Practices

### Log Levels

| Level | When | Examples |
|-------|------|----------|
| **ERROR** | Unexpected failure requiring investigation | Unhandled exception, data corruption, external service hard failure |
| **WARN** | Expected failure handled gracefully | Rate limit hit, cache miss fallback, deprecated API called |
| **INFO** | Significant business/operational events | Request completed, job started/finished, config loaded |
| **DEBUG** | Diagnostic detail for troubleshooting | SQL queries, cache hits, intermediate state |

Rules:
- ERROR means "wake someone up" -- if it doesn't, it's WARN
- INFO should tell the story of what the system did, not how
- DEBUG should never appear in production log aggregation by default
- Never log at ERROR for expected conditions (404s, validation failures)

### Structured Format

Always use structured (JSON) logging in production. Never `print()` or string-interpolated messages.

```
GOOD: {"level":"info","msg":"payment processed","user_id":"u_123","amount_cents":4999,"duration_ms":142,"trace_id":"abc"}
BAD:  INFO: Payment of $49.99 processed for user u_123 in 142ms
```

### Required Fields (every log line)

| Field | Purpose |
|-------|---------|
| `timestamp` | ISO 8601 with timezone |
| `level` | Log level |
| `msg` | Human-readable description |
| `service` | Service name |
| `trace_id` | Correlation with distributed traces |
| `request_id` | Request-scoped correlation |

### Context Enrichment

Add context at the right level -- don't repeat in every log call:
- **Request middleware**: inject `request_id`, `user_id`, `trace_id` into logger context
- **Service init**: inject `service`, `version`, `environment`
- **Per-call**: only add fields specific to that log event

### What NOT to Log

- PII (emails, names, phone numbers, IPs) -- hash or redact
- Secrets (tokens, passwords, API keys) -- never, even at DEBUG
- Full request/response bodies -- log selectively, redact sensitive fields
- High-frequency success paths without sampling -- kills storage budget

### Log Correlation

Every log line must include `trace_id` for correlation with distributed traces. Inject trace context from OpenTelemetry or your tracing library into the logger at request boundaries.

## Metrics Design

### Framework Selection

| Framework | When |
|-----------|------|
| **RED** (Rate, Errors, Duration) | Request-driven services (APIs, web servers) |
| **USE** (Utilization, Saturation, Errors) | Resource-scoped (CPU, memory, disk, connections) |
| **Golden Signals** (Latency, Traffic, Errors, Saturation) | General-purpose, covers both |

Pick ONE framework per service type. Don't mix.

### Metric Types

| Type | Use For | Example |
|------|---------|---------|
| **Counter** | Cumulative totals | `http_requests_total`, `errors_total` |
| **Histogram** | Latency distributions | `http_request_duration_seconds` |
| **Gauge** | Current values | `active_connections`, `queue_depth` |

Rules:
- Always histograms over summaries for latency -- histograms are aggregatable across instances
- Default histogram buckets: `[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]` seconds
- Exclude health/metrics endpoints from SLI calculations
- Use recording rules for any query in alerts or dashboards

### Label Cardinality

Label cardinality kills metric backends. Rules:
- **Never** use user IDs, request IDs, UUIDs, or unbounded values as labels
- **Safe labels**: method, status_code, endpoint (grouped), service, environment
- **Max ~10 unique values** per label dimension
- If you need high-cardinality breakdowns, use tracing or logs, not metrics

### Instrumentation Checklist

For every service, instrument at minimum:
- Request rate (counter by method, status, endpoint)
- Request duration (histogram by method, endpoint)
- Error rate (counter by method, error_type)
- Active connections / in-flight requests (gauge)
- Dependency call duration + error rate (per downstream service)
- Queue depth and processing lag (if async)

## Distributed Tracing

### When to Add Traces

- All cross-service calls (HTTP, gRPC, message queues)
- Database queries (as child spans)
- External API calls
- Significant internal operations (>10ms or business-critical)
- Async job dispatch and execution

### Sampling Strategy

| Environment | Strategy |
|-------------|----------|
| Dev/staging | 100% sampling |
| Production <1k rps | 10-50% probabilistic |
| Production >10k rps | 1% probabilistic or rate-limit ~100 traces/sec |

Always:
- Use `ParentBased` sampler so child spans follow parent's decision
- Force-sample all errors and high-latency requests regardless of probabilistic rate
- Use tail-based sampling at the collector when possible (sample interesting traces after completion)

### Context Propagation

- Use W3C `traceparent` header for new systems (not B3 or Jaeger-native)
- Inject `trace_id` into structured logs for log-trace correlation
- Propagate context explicitly through async boundaries (queues, event buses, cron jobs)
- When context is lost (e.g., crossing a message queue), create a new trace and link to parent via `trace_id` in message metadata

### Span Best Practices

- Name spans as `<component>.<operation>` (e.g., `db.query`, `http.request`, `queue.publish`)
- Add attributes: `db.statement`, `http.method`, `http.status_code`, `messaging.destination`
- Set span status to ERROR on failures with error message
- Keep span count reasonable -- 50-200 spans per trace is normal; >1000 indicates over-instrumentation

### Collector Architecture

Always send telemetry via OpenTelemetry Collector, never direct from app to backend:
```
App -> OTel SDK -> OTel Collector -> Backend (Tempo/Jaeger/vendor)
```

Benefits: buffering, retry, sampling decisions, multi-backend fanout, protocol translation.

## SLOs and Alerting

### SLO Design

1. Classify service tier first (determines targets):

| Tier | Availability | Latency P99 | Error Budget/Month |
|------|-------------|-------------|-------------------|
| Critical (payment, auth) | 99.95% | 100ms | 21.6 min |
| Essential (search, catalog) | 99.9% | 500ms | 43.2 min |
| Standard (recommendations) | 99.5% | 1s | 3.6 hr |
| Best Effort (batch, reporting) | 99.0% | 2s | 7.2 hr |

2. Define SLIs (what you measure):
   - **Availability SLI**: proportion of successful requests (exclude 4xx client errors)
   - **Latency SLI**: proportion of requests faster than threshold

3. Start conservative: 99.0% for 1 month baseline, then tighten progressively. Never set SLO tighter than current measured reliability.

### Error Budget Policy

| Budget Remaining | Action |
|-----------------|--------|
| >50% | Normal development velocity |
| 10-50% | Postpone risky changes |
| 1-10% | Feature freeze, reliability only |
| 0% | Full stop, postmortem required |

### Alert Design

Rules:
- **Alert on symptoms, not causes** -- alert on error rate, not "pod restarted"
- Every alert MUST have a runbook link
- Severity: `critical` (pages on-call), `warning` (creates ticket), `info` (dashboard only)
- `for:` duration minimums: critical >= 2min, warning >= 5min, info >= 15min

### Burn Rate Alerts (preferred SLO alerting)

| Alert | Burn Rate | Window | Action |
|-------|-----------|--------|--------|
| Fast burn | 14.4x | 1h + 5m confirmation | Page on-call |
| Slow burn | 3x | 6h + 30m confirmation | Create ticket |

Multi-window burn rate is the only correct SLO alerting pattern.

## Analytics Instrumentation

### Event Taxonomy

Use `Object Action` format in past tense. Pick one casing and enforce:
```
GOOD: Account Created, Subscription Started, Report Exported
BAD:  clickedButton, Create Account, user_did_thing
```

### Tracking Plan Discipline

- Define events in a typed schema (TypeScript types, JSON Schema, or tracking plan doc)
- CI lint for unregistered event names
- Review tracking changes in PRs like API changes
- Start with 10-15 events mapping to core funnel; add only when someone asks an unanswerable question
- Remove events unqueried for 90 days

### Client vs Server Tracking

| Track On | What |
|----------|------|
| **Server-side** | Revenue, signups, conversions, subscription changes -- anything ad blockers would drop |
| **Client-side** | UI interactions, page views, feature usage, navigation patterns |

Reconcile client/server counts weekly. Expect 10-30% client-side drop from ad blockers.

### Analytics Anti-Patterns

- **PII leakage**: allowlist permitted properties per event; middleware strips non-allowed fields
- **Cardinality explosion**: never group by UUIDs or free text; bucket continuous values
- **Over-tracking**: every click tracked, 500 event types, nobody queries 450 -- storage bill explodes
- **Naming drift**: enforce typed schema + CI checks; "Button Clicked" vs "button_clicked" vs "btn_click" is tech debt

## Dashboard Design

### Hierarchy

```
L1: Service overview (4-6 golden signals, trend indicators, alerts)
L2: Component drilldown (per-endpoint, per-dependency metrics)
L3: Investigation (traces, logs, detailed breakdowns)
```

### Dashboard Rules

- Every dashboard must use template variables (environment, service, time range)
- No more than 12 panels per dashboard -- split into drilldowns
- Include annotation markers for deployments and incidents
- Dashboard-as-code (Grafana JSON, Terraform, or provider API) -- never click-ops only
- Every panel must have a description explaining what it shows and what "bad" looks like

## OpenTelemetry Integration Checklist

For greenfield or migration to OTel:

1. **SDK setup**: install OTel SDK for your language, configure exporters (OTLP)
2. **Auto-instrumentation**: enable for HTTP, DB, messaging frameworks first
3. **Custom spans**: add for business-critical operations
4. **Custom metrics**: add RED metrics per service
5. **Log bridge**: connect structured logger to OTel log exporter (or inject trace context into logs)
6. **Collector**: deploy OTel Collector as sidecar or gateway
7. **Semantic conventions**: follow OTel semantic conventions for attribute names
8. **Resource attributes**: set `service.name`, `service.version`, `deployment.environment`
