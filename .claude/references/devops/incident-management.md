# Incident Management

## Severity Framework

| Severity | Impact | Response Time | Example |
|----------|--------|---------------|---------|
| **SEV1** | Complete outage, data loss | 5 min | Production down |
| **SEV2** | Major degradation | 15 min | Critical feature broken |
| **SEV3** | Minor impact | 2 hours | Non-critical bug |
| **SEV4** | Minimal impact | Next business day | Cosmetic issue |

## Incident Command Roles

| Role | Responsibility | Handoff Trigger |
|------|----------------|-----------------|
| **Incident Commander (IC)** | Owns resolution, delegates work, makes decisions | Shift change or 2+ hours in |
| **Communications Lead** | Status updates to stakeholders every 15 min (SEV1) / 30 min (SEV2) | IC assigns when external impact |
| **Tech Lead** | Drives technical investigation and fix | IC assigns for complex incidents |
| **Scribe** | Maintains real-time timeline in incident channel | IC assigns for SEV1/SEV2 |

### IC Decision Authority
- IC can roll back without approval
- IC can page anyone in the org
- IC cannot make permanent architecture changes (post-incident only)

## Runbook Structure

### Triage Decision Table

| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| All requests failing | Service down | Rollback |
| High latency | Database/dependency | Check connections |
| Partial failures | Code bug | Feature flag disable |
| Spike in errors | Traffic surge | Scale up |

### Escalation Triggers

**Immediate:** SEV1, data breach, unable to diagnose within 30 min
**Consider:** Spans multiple teams, requires expertise you lack, uncertain about next steps

| Condition | Escalate To |
|-----------|-------------|
| > 15 min unresolved SEV1 | Engineering Manager |
| Data breach suspected | Security Team |
| Customer communication needed | Support Lead |

### Auto-Escalation Triggers

| Condition | Action | Timeline |
|-----------|--------|----------|
| Impact doubling (users, error rate) | Escalate severity by 1 | Immediate |
| No root cause identified | Page senior engineer | After 30 min |
| No IC response to page | Page backup IC + manager | After 5 min |
| Customer-facing SEV1 with no status update | Alert Comms Lead | After 10 min |
| Incident crosses team boundaries | IC pages second team lead | Immediate |

## On-Call Handoff

### Required Components

| Component | Purpose |
|-----------|---------|
| Active Incidents | What's currently broken |
| Ongoing Investigations | Issues being debugged |
| Recent Changes | Deployments, configs |
| Known Issues | Workarounds in place |
| Upcoming Events | Maintenance, releases |

### Handoff Timing

30 min overlap: outgoing writes handoff (15 min) + sync call (15 min). Incoming reviews + verifies alerting.

## Postmortem Writing

### Blameless Culture

| Blame-Focused | Blameless |
|---------------|-----------|
| "Who caused this?" | "What conditions allowed this?" |
| Punish individuals | Improve systems |
| Hide information | Share learnings |

### Timeline

```
Day 0: Incident occurs
Day 1-2: Draft postmortem
Day 3-5: Postmortem meeting (60 min)
Day 5-7: Finalize, create tickets
Week 2+: Action item completion
Quarterly: Review patterns
```

### Required Sections

1. **Executive Summary** -- 1-2 sentences: what, impact, resolution
2. **Timeline (UTC)** -- timestamped events
3. **Root Cause (5 Whys)** -- keep asking "why" until you hit a systemic issue
4. **Detection** -- what worked, what didn't
5. **Response** -- what worked, what could improve
6. **Lessons Learned** -- went well, went wrong, got lucky
7. **Action Items** -- priority, owner, due date, ticket (always concrete)
