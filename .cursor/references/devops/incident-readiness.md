# Incident Readiness

## On-Call Rotation Design

| Factor | Recommendation |
|--------|----------------|
| **Minimum team size** | 2 primary + 2 secondary (minimum); 3+ primary preferred |
| **Shadow period** | New members shadow primary for 2 weeks before taking shifts |
| **Rotation length** | 1 week (recommended), 2 weeks max — longer causes burnout |
| **Burnout threshold** | >2 pages/week average = team too small or system too noisy |
| **Compensation** | Time-off (1 day per on-call week) or pay differential |
| **Handoff day** | Mid-week (Tuesday/Wednesday) — avoids Monday rush and Friday context loss |
| **Escalation path** | Primary → Secondary → Engineering Manager → VP (each has 5 min to ack) |

## On-Call Expectations

| Severity | Ack Time | Response Time | Escalation |
|----------|----------|---------------|------------|
| **SEV1** | 5 min | 15 min active investigation | Auto-escalate if no ack in 5 min |
| **SEV2** | 15 min | 30 min active investigation | Auto-escalate if no ack in 15 min |
| **SEV3** | 1 hour | 4 hours | Manual escalation |
| **SEV4** | Next business day | Next business day | No escalation |

## Game Day Methodology

### Planning Checklist
- [ ] Scope defined (which system, which failure mode)
- [ ] Blast radius limited (single service, non-peak hours)
- [ ] Rollback plan tested (can you undo the injection in <1 min?)
- [ ] Monitoring dashboard identified and shared
- [ ] All participants briefed on kill switch location
- [ ] Stakeholders notified (at minimum: on-call, manager)

### Execution Steps
1. **Announce** — "Game day starting. Injecting [failure] into [system] at [time]."
2. **Inject** — Apply the failure (kill process, add latency, drop packets)
3. **Observe** — Did alerts fire? How fast? Did dashboards reflect the issue?
4. **Recover** — Use the kill switch or let auto-recovery work
5. **Debrief** — What worked? What surprised us? What do we fix?

### Progressive Complexity

| Level | Focus | Example |
|-------|-------|---------|
| **1** | Known failure modes | Kill a single pod, restart a database |
| **2** | Unknown interactions | Add 500ms latency between two services |
| **3** | Multi-failure cascades | Network partition + increased traffic |

### Safety Rules
- Production requires VP-level approval
- Always have a tested kill switch
- Start with smallest possible blast radius
- Never inject during peak traffic or change freezes
- Have rollback tested before injection begins

## Incident Analytics

| Metric | Definition | Target (SEV1) | Target (SEV2) |
|--------|------------|---------------|----------------|
| **MTTD** (Mean Time to Detect) | Alert fired → human aware | <5 min | <15 min |
| **MTTR** (Mean Time to Resolve) | Detection → resolution | <1 hour | <4 hours |
| **MTBF** (Mean Time Between Failures) | Incident → next incident | >30 days | >14 days |
| **Pages per shift** | Total pages per on-call week | <5 | — |
| **False positive rate** | Non-actionable alerts / total | <20% | <20% |

**Deploy correlation:** Track all incidents that occur within 2 hours of a deploy. If >30% correlate, your deploy pipeline needs safety improvements (canary, automated rollback).

## Alert Quality

| Indicator | Target | Action If Missed |
|-----------|--------|------------------|
| Actionable alert rate | >80% | Tune thresholds, consolidate noisy alerts |
| Pages per person per day | <5 | Silence low-severity, fix root causes |
| Alert review cadence | Monthly | Delete alerts nobody acts on |
| Runbook coverage | 100% of alerts have runbooks | Block new alerts without runbook links |

## Readiness Checklist

Before starting an on-call shift, verify:

- [ ] VPN, kubectl, database, log aggregator access working
- [ ] PagerDuty/Opsgenie shows you as primary on-call
- [ ] Phone/laptop notifications enabled and tested
- [ ] Test alert received and acknowledged
- [ ] Reviewed incidents from past 2 weeks
- [ ] Runbooks for top 5 alert types reviewed
- [ ] Escalation contacts confirmed and reachable
- [ ] Know where the kill switches are for critical systems
