# SRE Practices

## Error Budget Policy

**Formula:** Error budget = 1 − SLO

| SLO | Budget/Month | Budget/Quarter |
|-----|-------------|----------------|
| 99.99% | 4.3 min | 13 min |
| 99.95% | 21.6 min | 65 min |
| 99.9% | 43.2 min | 130 min |
| 99.5% | 3.6 hours | 10.8 hours |

### Enforcement Policy

| Budget Remaining | Policy | Actions |
|------------------|--------|---------|
| **>50%** | Ship freely | Normal development velocity |
| **25–50%** | Caution | No risky launches (new infra, major migrations) |
| **10–25%** | Feature freeze | Only reliability and bug-fix work ships |
| **<10%** | All hands reliability | Entire team focused on stability |
| **Exhausted** | Full stop | Only reliability work ships; postmortem required for any further incidents |

**Review cadence:** Weekly error budget check in team standup. Monthly trend review with engineering leadership.

## Toil Elimination

**Toil definition:** Manual, repetitive, automatable work that scales linearly with service growth and has no enduring value.

| Toil Indicator | Example | Automation Approach |
|---|---|---|
| Manual deploys | SSH + run scripts | CI/CD pipeline |
| Certificate rotation | Calendar reminder + manual renewal | cert-manager / ACME auto-renewal |
| Capacity requests | Ticket → manual scaling | Auto-scaling policies |
| Log investigation for known issues | Grep logs → restart service | Self-healing + auto-remediation |
| On-call handoff docs | Write from memory each week | Auto-generated from incident/deploy logs |

**Target:** <50% of on-call engineer time spent on toil. Measure weekly. If above target, prioritize automation over feature work.

**Cycle:** Identify (log toil for 2 weeks) → Measure (rank by time spent) → Prioritize (highest time first) → Automate → Verify (time spent decreased)

## Capacity Planning

### Forecasting
- Use trailing 90-day growth rate as baseline
- Plan for **2× current peak** headroom
- Run load tests monthly at 2× peak

### Scaling Triggers

| Signal | Threshold | Action |
|--------|-----------|--------|
| CPU utilization | >70% sustained 5 min | Scale horizontally |
| Memory utilization | >80% | Investigate (leak?) then scale |
| Queue depth | Growing over 10 min | Scale consumers |
| p99 latency | >SLO target | Scale or optimize hot path |
| Disk usage | >75% | Expand or archive |
| Connection pool | >80% utilized | Increase pool or optimize queries |

## Progressive Deployment

| Stage | Traffic | Bake Time | Rollback Trigger |
|-------|---------|-----------|------------------|
| **Canary** | 1–5% | 15 min | Error rate >2× baseline |
| **Ring 1** | 25% | 30 min | Latency p99 >2× baseline |
| **Ring 2** | 50% | 30 min | Crash rate increase |
| **Full** | 100% | 24 hr monitoring | Any SLO breach |

### Auto-Rollback Triggers
- Error rate >2× baseline for 5 min
- Latency p99 >2× baseline for 5 min
- Crash/restart loop detected (>3 restarts in 10 min)
- Health check failures >10%

**Rule:** Every deploy must be revertible within 5 minutes. If rollback requires a migration, the deploy needs a separate rollback plan reviewed before merge.

## SLO Targets by Service Tier

| Tier | Examples | Availability | Latency p50 | Latency p99 | Error Budget/Month |
|------|----------|--------------|-------------|-------------|-------------------|
| **Tier 1** (Revenue-critical) | Checkout, auth, API gateway | 99.95% | <50ms | <200ms | 21.6 min |
| **Tier 2** (Business-critical) | Search, notifications, admin | 99.9% | <100ms | <500ms | 43.2 min |
| **Tier 3** (Internal tools) | CI/CD, dev portals, analytics | 99.5% | <500ms | <2s | 3.6 hr |

**Rule:** Every service must declare its tier in service metadata. Tier determines on-call expectations, error budget policy, and deployment rigor.
