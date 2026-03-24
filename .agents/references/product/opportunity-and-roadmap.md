# Opportunity Assessment & Roadmap

## RICE Scoring

Score = (Reach × Impact × Confidence) / Effort

| Dimension | 1 | 2 | 3 | 4 | 5 |
|-----------|---|---|---|---|---|
| **Reach** | <100 users/quarter | 100-500 | 500-2K | 2K-10K | >10K |
| **Impact** | Minimal improvement | Minor improvement | Moderate — notable UX gain | High — solves key pain | Massive — unlocks new segment |
| **Confidence** | Gut feel only | Anecdotal feedback | Survey/interview data | A/B test or prototype data | Production data proves it |
| **Effort** | >3 months | 2-3 months | 1-2 months | 2-4 weeks | <2 weeks |

### Example Scored Opportunities

| Opportunity | R | I | C | E | Score | Decision |
|-------------|---|---|---|---|-------|----------|
| Bulk user invite | 4 | 4 | 4 | 3 | 21.3 | Now |
| Dark mode | 3 | 2 | 3 | 2 | 9.0 | Later |
| SSO integration | 3 | 5 | 5 | 4 | 18.8 | Next |
| Custom dashboards | 2 | 3 | 2 | 5 | 2.4 | Not building |

## Now / Next / Later Roadmap

| Tier | Timeframe | Required Artifact | Commitment Level |
|------|-----------|-------------------|------------------|
| **Now** | This sprint/month | Full PRD + acceptance criteria | Committed — staffed and in progress |
| **Next** | Next quarter | Problem statement + evidence + RICE score | Planned — scoped but not started |
| **Later** | Backlog | 1-line description | Aspirational — revisit quarterly |

**Rules:**
- Now items cannot exceed team capacity (WIP limit)
- Moving from Next → Now requires a PRD
- Later items are culled quarterly — delete anything with no champion
- Stakeholders can see all tiers; only Now is a promise

## North Star Metric

Pick ONE metric that captures the core value your product delivers:

| Product Type | North Star | Why |
|---|---|---|
| Marketplace | GMV (Gross Merchandise Value) | Measures both supply and demand health |
| SaaS B2B | Weekly Active Users | Habitual use = retention = revenue |
| Content/Media | Time Spent (quality-adjusted) | Engagement drives ad revenue and retention |
| Dev Tool | Tasks Completed per Week | Measures actual productivity delivered |
| Fintech | Transaction Volume | Core value = moving money |

**Test:** If this metric goes up and everything else stays flat, is the business healthier? If yes, it's your North Star.

## "Not Building" List

Explicitly document rejected features to prevent re-litigation:

| Feature | Why Not | Revisit If |
|---------|---------|------------|
| Mobile app | <5% of users on mobile; web-responsive is sufficient | Mobile usage exceeds 20% |
| AI chatbot | No evidence users want conversational UX | Competitor ships one successfully |
| Multi-language | All current customers are English-speaking | International expansion begins |

## GTM Brief

### Launch Checklist

- [ ] Feature flag ready for staged rollout
- [ ] Documentation / changelog updated
- [ ] Support team briefed (FAQ + escalation path)
- [ ] Monitoring dashboard configured (errors, latency, adoption)
- [ ] Rollback procedure tested
- [ ] Success metrics baseline captured

### Rollback Triggers

| Signal | Threshold | Action |
|--------|-----------|--------|
| Error rate | >2× baseline | Auto-rollback via feature flag |
| Latency p99 | >2× baseline | Investigate; rollback if no fix in 30 min |
| User complaints | >10 in first hour | Disable for new users, investigate |
| Data integrity | Any corruption | Immediate rollback + incident |

### Post-Launch Metric Targets

| Timeframe | Metric | Target |
|-----------|--------|--------|
| **7 days** | Feature adoption rate | >10% of eligible users |
| **30 days** | Task completion rate | >60% of users who start |
| **90 days** | Impact on North Star | Measurable positive movement |

## Operating Principles

1. **Problem-first**: Evidence before solution. No PRD without data.
2. **Smallest viable scope**: Ship the minimum that tests the hypothesis.
3. **Measure before scaling**: Instrument before you build. Data before conviction.
4. **Kill what doesn't move metrics**: Sunset features with <5% adoption after 90 days.
5. **Say no by default**: Every yes is a no to something else. Defend focus.
