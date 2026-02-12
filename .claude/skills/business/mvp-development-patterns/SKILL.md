---
name: mvp-development-patterns
description: Rapid prototyping patterns, scope-cutting frameworks, build-measure-learn loops, and technical debt management for early-stage startups.
---

# MVP Development Patterns

## When to Use This Skill

Use when building a first product version, deciding build vs buy, cutting scope to ship faster, or managing technical debt in early-stage codebases. Applies from idea validation through first paying customers.

## Build vs Buy vs Open-Source

| Factor | Build Custom | Buy (SaaS) | Open-Source + Host |
|---|---|---|---|
| Time to integrate | Weeks-months | Hours-days | Days-weeks |
| Upfront cost | Engineering time | Subscription | Engineering + infra |
| Ongoing cost | Maintenance burden | Scales with usage | Maintenance + infra |
| Customization | Full control | Limited to API/config | Fork and modify |
| When to choose | Core differentiator | Commodity function | Need control + budget |
| Examples | Your ML pipeline | Auth (Auth0), email (SendGrid) | DB, search, queues |

### Decision Rule

If the feature is **not** your core value prop, buy it. If you would be embarrassed showing investors you spent 3 weeks building it, buy it. Build only what differentiates you.

**Common buy**: Auth, payments, email/SMS, error tracking, logging, feature flags, analytics.
**Common build**: Core product logic, domain-specific data pipelines, custom ML models.

## Scope-Cutting Framework

### The 3-Question Filter

For every feature in the backlog:

1. **Does a user need this to get value on day one?** No -> cut it.
2. **Can we validate the hypothesis without building it?** Yes -> fake it first.
3. **Will <10% of users use this in month one?** Yes -> defer it.

### Scope Levels

```
Level 0 (Wizard of Oz):  Fake it manually behind the scenes
Level 1 (Concierge MVP): Semi-automated, human-in-the-loop
Level 2 (Duct Tape MVP): Works but ugly, hardcoded paths
Level 3 (Real MVP):      Automated, handles edge cases
Level 4 (Product):       Polished, scalable, monitored
```

Ship Level 2. Iterate to Level 3 only after validation. Most features never need Level 4.

## Feature Flag Gated MVP

```typescript
interface MVPConfig {
  flags: Record<string, {
    enabled: boolean;
    allowList: string[];  // Early access user IDs
    level: 0 | 1 | 2 | 3 | 4;
  }>;
}

const mvpConfig: MVPConfig = {
  flags: {
    "ai-summary": {
      enabled: true,
      allowList: ["user_alpha1", "user_alpha2"],
      level: 2,  // Duct tape: works but slow, no caching
    },
    "team-sharing": {
      enabled: false,
      allowList: [],
      level: 0,  // Wizard of Oz: founder manually shares via email
    },
    "export-pdf": {
      enabled: true,
      allowList: [],
      level: 1,  // Concierge: queues request, team generates manually
    },
  },
};

function canAccess(feature: string, userId: string): boolean {
  const flag = mvpConfig.flags[feature];
  if (\!flag || \!flag.enabled) return false;
  if (flag.allowList.length === 0) return true;
  return flag.allowList.includes(userId);
}
```

## Analytics-First Architecture

Instrument before you build. Every MVP feature should emit events from day one.

```typescript
class Analytics {
  private providers: AnalyticsProvider[] = [];
  constructor(providers: AnalyticsProvider[]) { this.providers = providers; }

  track(event: { name: string; properties: Record<string, string | number | boolean> }): void {
    const enriched = {
      ...event,
      timestamp: new Date(),
      properties: {
        ...event.properties,
        app_version: process.env.APP_VERSION ?? "unknown",
        environment: process.env.NODE_ENV ?? "development",
      },
    };
    for (const p of this.providers) p.send(enriched).catch(console.error);
  }

  trackMVPFunnel(step: string, userId: string, meta?: Record<string, string>): void {
    this.track({ name: `mvp_funnel_${step}`, properties: { userId, step, ...meta } });
  }
}

// Validate your hypothesis with data
analytics.trackMVPFunnel("landed", userId);
analytics.trackMVPFunnel("signed_up", userId, { source: "google" });
analytics.trackMVPFunnel("completed_onboarding", userId);
analytics.trackMVPFunnel("first_value_moment", userId);
analytics.trackMVPFunnel("returned_day1", userId);
analytics.trackMVPFunnel("converted_paid", userId, { plan: "pro" });
```

## Modular Monolith Starter

Start with a monolith, but structure it so extraction is cheap later.

```typescript
// Module boundary enforcement: modules communicate through interfaces
// src/modules/core/core.service.ts
import type { BillingService } from "../billing/billing.types";

export class CoreService {
  constructor(private billing: BillingService) {}

  async processItem(userId: string, item: Item): Promise<Result> {
    const canProcess = await this.billing.checkQuota(userId);
    if (\!canProcess) throw new QuotaExceededError(userId);
    const result = await this.doWork(item);
    await this.billing.recordUsage(userId, 1);
    return result;
  }
}
// When you extract billing to a service:
// 1. Implement BillingService interface as HTTP client
// 2. Swap injection
// 3. Zero changes to CoreService
```

### Directory Structure

```
src/
  modules/           # Domain boundaries (future service boundaries)
    auth/            # auth.service.ts, auth.routes.ts, auth.types.ts
    billing/         # billing.service.ts, billing.routes.ts, billing.types.ts
    core/            # core.service.ts, core.routes.ts, core.types.ts
  shared/            # Cross-cutting: database.ts, logger.ts, config.ts
  app.ts             # Composes modules
  server.ts          # Entry point
```

## Build-Measure-Learn Loop

```
Week 1: Build hypothesis + minimal implementation (Level 2)
        - Define success metric (1 number)
        - Instrument analytics events
        - Ship to 10-50 users

Week 2: Measure behavior
        - Check funnel completion rates
        - Do 5 user interviews
        - Identify biggest drop-off

Week 3: Learn + decide
        - Pivot: metric did not move -> try different approach
        - Persevere: metric moved -> invest in Level 3
        - Kill: no signal at all -> move on
```

### Choosing Boring Technology

| Decision | Boring (Recommended) | Shiny (Avoid Unless Core) |
|---|---|---|
| Language | TypeScript, Python | Rust, Elixir, Zig |
| Database | PostgreSQL | CockroachDB, Fauna, SurrealDB |
| Queue | Redis + BullMQ | Kafka, Pulsar |
| Cache | Redis | Memcached cluster, Dragonfly |
| Search | PostgreSQL full-text | Elasticsearch (until >1M docs) |
| Hosting | Railway, Render, Fly.io | Kubernetes |
| Frontend | Next.js, Remix | Your own framework |

**Rule**: You get 3 innovation tokens. Spend them on your core differentiator, not infrastructure.

## Gotchas and Anti-Patterns

### Premature Scaling
- **Problem**: Building for 1M users when you have 10. Kubernetes, microservices, multi-region from day one.
- **Fix**: A single $50/mo server handles 100K+ requests/day. Scale when you have the problem. Vertical scaling buys 6-12 months every time.
### Over-Engineering Auth
- **Problem**: 3 weeks building custom auth with MFA, SSO, passwordless, account recovery.
- **Fix**: Use Auth0, Clerk, or Supabase Auth. Hours, not weeks. Auth is never your differentiator.
### Building Before Validating
- **Problem**: 3 months building, zero users talked to. Ship day arrives, nobody wants it.
- **Fix**: Talk to 10 potential users before writing code. Landing page with waitlist. Wizard of Oz the first 5 customers. Code is the most expensive way to validate.
### Choosing Trendy Tech
- **Problem**: New database from Hacker News. Undocumented edge cases at 2 AM with no Stack Overflow answers.
- **Fix**: Pick tech with large communities, extensive docs, 5+ years production use. Startup risk should be in the product, not infrastructure.
### Infinite Polish Loop
- **Problem**: Two weeks on animations and edge cases before anyone has used the feature.
- **Fix**: Ship ugly. If users complain about looks but use it, polish. If nobody uses it, you saved two weeks.
