---
name: analytics-instrumentation
description: Event tracking architecture, funnel design, analytics provider integration, event taxonomy, and data governance for product analytics.
---

# Analytics Instrumentation

## When to Use This Skill

Use when designing event tracking systems, integrating analytics providers, building funnels, or establishing event taxonomy standards. Applies to greenfield instrumentation and fixing messy existing tracking.

## Analytics Provider Selection

| Criteria | Mixpanel | Amplitude | PostHog | Segment (CDP) |
|---|---|---|---|---|
| Free tier | 20M events/mo | 50K MTU | 1M events/mo (self-host unlimited) | 1,000 MTU |
| Strength | Funnel/retention | Behavioral cohorts | All-in-one (analytics+flags+replay) | Data routing |
| Weakness | Expensive at scale | Complex pricing | UI less polished | Still need analytics tool |
| Self-host | No | No | Yes (full feature) | No |
| Best for | PLG metrics | Enterprise analytics | Startups wanting control | Multi-tool stack |

### Decision Rule

- **Bootstrapped / < $1M ARR**: PostHog. Analytics, feature flags, session replay, A/B testing in one tool.
- **Funded startup / PLG**: Mixpanel or Amplitude. Better funnel analysis UIs.
- **Multiple analytics tools**: Segment as router -> Mixpanel + data warehouse.
- **Privacy-first / EU**: PostHog self-hosted. Full data residency control.

## Event Tracking Wrapper

```typescript
type EventProperties = Record<string, string | number | boolean | null>;

interface AnalyticsProvider {
  identify(userId: string, traits: EventProperties): void;
  track(event: string, properties: EventProperties): void;
  page(name: string, properties?: EventProperties): void;
  reset(): void;
}

class AnalyticsClient implements AnalyticsProvider {
  constructor(private providers: AnalyticsProvider[]) {}

  identify(userId: string, traits: EventProperties): void {
    for (const p of this.providers) p.identify(userId, traits);
  }

  track(event: string, properties: EventProperties): void {
    const enriched: EventProperties = {
      ...properties,
      timestamp: new Date().toISOString(),
      session_id: this.getSessionId(),
      page_url: typeof window \!== "undefined" ? window.location.href : null,
    };
    for (const p of this.providers) p.track(event, enriched);
  }

  page(name: string, properties?: EventProperties): void {
    for (const p of this.providers) p.page(name, properties ?? {});
  }

  reset(): void {
    for (const p of this.providers) p.reset();
  }

  private getSessionId(): string {
    return sessionStorage?.getItem("analytics_session_id") ?? "unknown";
  }
}
```

### Provider Implementation (PostHog)

```typescript
import posthog from "posthog-js";

export class PostHogProvider implements AnalyticsProvider {
  constructor(apiKey: string, host: string) {
    posthog.init(apiKey, { api_host: host, capture_pageview: false });
  }
  identify(userId: string, traits: EventProperties): void { posthog.identify(userId, traits); }
  track(event: string, properties: EventProperties): void { posthog.capture(event, properties); }
  page(name: string, properties: EventProperties): void {
    posthog.capture("$pageview", { page_name: name, ...properties });
  }
  reset(): void { posthog.reset(); }
}
```

## Event Taxonomy

### Naming Convention

Use `Object Action` format in past tense. Pick one casing and enforce it.

```
GOOD (Object Action, past tense):       BAD (inconsistent):
  Account Created                          clickedButton
  Subscription Started                     Create Account
  Report Exported                          user_did_thing
  Search Performed                         report.export
  Invite Sent                              btnClick
```

### Standard Event Schema

```typescript
type CoreEvents = {
  "Account Created": {
    signup_method: "email" | "google" | "github";
    referral_source: string | null;
  };
  "Feature Used": {
    feature_name: string;
    feature_category: string;
    is_first_use: boolean;
  };
  "Subscription Started": {
    plan: "free" | "pro" | "enterprise";
    billing_cycle: "monthly" | "annual";
    trial: boolean;
    mrr_cents: number;
  };
};

function trackEvent<K extends keyof CoreEvents>(
  event: K, properties: CoreEvents[K],
): void {
  analytics.track(event, properties as EventProperties);
}

// Compile-time error if properties do not match schema
trackEvent("Account Created", { signup_method: "google", referral_source: "producthunt" });
```

## Funnel Definition

```typescript
interface Funnel {
  name: string;
  steps: { event: string; description: string; expectedDropoff?: number }[];
}

const signupFunnel: Funnel = {
  name: "Signup to Activation",
  steps: [
    { event: "Landing Page Viewed", description: "Hit the website" },
    { event: "Signup Started", description: "Clicked signup CTA", expectedDropoff: 60 },
    { event: "Account Created", description: "Completed form", expectedDropoff: 20 },
    { event: "Onboarding Started", description: "Entered onboarding", expectedDropoff: 10 },
    { event: "First Value Moment", description: "Core action completed", expectedDropoff: 30 },
    { event: "Returned Day 1", description: "Came back next day", expectedDropoff: 50 },
  ],
};

function checkFunnelHealth(funnel: Funnel, actualDropoffs: number[]): string[] {
  const alerts: string[] = [];
  funnel.steps.forEach((step, i) => {
    const expected = step.expectedDropoff ?? 0;
    if ((actualDropoffs[i] ?? 0) > expected + 10)
      alerts.push(`${step.event}: ${actualDropoffs[i]}% dropoff (expected ${expected}%)`);
  });
  return alerts;
}
```

## User Identification

```typescript
function handleSignup(anonymousId: string, userId: string, traits: EventProperties): void {
  analytics.alias(anonymousId, userId);
  analytics.identify(userId, { ...traits, created_at: new Date().toISOString() });
  analytics.track("Account Created", traits);
}

function trackServerEvent(userId: string, event: string, properties: EventProperties): void {
  serverAnalytics.track({
    userId, event,
    properties: { ...properties, source: "server", service: process.env.SERVICE_NAME },
    context: { ip: null, active: false },  // Do not forward server IP or count as active
  });
}
```

## Property Standardization

```typescript
const standardize = {
  currency: (cents: number) => ({
    amount_cents: cents, amount_dollars: cents / 100, currency: "USD",
  }),
  user: (user: User) => ({
    user_id: user.id, user_plan: user.plan, user_created_at: user.createdAt.toISOString(),
  }),
  page: () => ({
    page_url: window.location.href, page_path: window.location.pathname,
    page_referrer: document.referrer || null,
  }),
};

analytics.track("Purchase Completed", {
  ...standardize.currency(4999),
  ...standardize.user(currentUser),
  ...standardize.page(),
  item_id: "prod_123",
});
```

## Gotchas and Anti-Patterns

### Tracking Plan Drift
- **Problem**: Devs add events ad-hoc. Names diverge (`Button Clicked`, `button_clicked`, `btn_click`). Properties inconsistent.
- **Fix**: Define events in typed schema. Auto-generate docs from types. CI lint for unregistered events. Review tracking in PRs like API changes.
### PII in Events
- **Problem**: Emails, names, IPs accidentally in event properties. Violates GDPR/CCPA. Cannot easily delete from analytics provider.
- **Fix**: Allowlist permitted properties per event. Middleware strips non-allowed fields. Never pass raw user objects. Quarterly PII audits.
### Cardinality Explosion
- **Problem**: High-cardinality values as properties (UUIDs, timestamps, free text). Queries slow, grouping useless.
- **Fix**: Bucket continuous values. Use IDs only for joins, never grouping. Cap enum values at <100 per property.
### Client vs Server-Side Discrepancies
- **Problem**: Client tracks "Purchase Started", server tracks "Purchase Completed." Ad blockers drop 10-30% of client events.
- **Fix**: Track revenue/signups/conversions server-side. Client-side for UI interactions only. Reconcile counts weekly.
### Over-Tracking
- **Problem**: Every click, hover, scroll tracked. 500 event types, nobody queries 450. Analytics bill explodes.
- **Fix**: Start with 10-15 events mapping to core funnel. Add only when someone asks an unanswerable question. Remove events unqueried for 90 days.
