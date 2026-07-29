---
name: j-observe
description: "Observability consultation -- logging, metrics, tracing, alerting, and analytics instrumentation. Use when designing observability systems, adding instrumentation, setting up dashboards, or defining SLOs. Do NOT use for CI/CD pipeline config (use /j-devops) or cloud infrastructure (use /j-cloud)."
argument-hint: "<question-or-task>"
model: sonnet
effort: medium
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

For deep observability design, you may delegate to the `devops-engineer` agent via the Task tool (it loads the `references/devops/` library incl. observability, sre-practices, incident-management). Verify its output before presenting.

Help with: $ARGUMENTS
