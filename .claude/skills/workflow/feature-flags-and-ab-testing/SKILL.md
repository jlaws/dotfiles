---
name: feature-flags-and-ab-testing
description: Feature flag systems, gradual rollout strategies, A/B test design, and statistical significance for product experiments.
---

# Feature Flags and A/B Testing

## When to Use This Skill

Use when implementing feature toggles, gradual rollouts, or controlled experiments. Applies to provider selection, flag lifecycle management, and experiment design with statistical rigor.

## Provider Selection

| Criteria | LaunchDarkly | Unleash | Flagsmith | Custom (DB/Config) |
|---|---|---|---|---|
| Cost at 10K MAU | ~$833/mo | Free (self-host) | Free tier | Infrastructure only |
| Cost at 1M MAU | ~$3,333/mo | Free (self-host) | ~$45/mo | Infrastructure only |
| Setup time | Hours | Half-day | Hours | Days-weeks |
| Targeting complexity | Advanced | Moderate | Moderate | Build it yourself |
| Edge evaluation | Yes | Via proxy | Via proxy | No |
| Audit trail | Built-in | Built-in | Built-in | Build it yourself |
| Best for | Funded startup, complex rules | Self-host preference, privacy | Budget-conscious teams | <5 flags, full control |

### Recommendation

Start with Flagsmith or Unleash if budget-constrained. Move to LaunchDarkly when targeting rules get complex or you need sub-50ms edge evaluation. Custom only if you have fewer than 5 flags and no rollout needs.

## Flag SDK Integration

### LaunchDarkly (TypeScript)

```typescript
import * as ld from "launchdarkly-node-server-sdk";

const client = ld.init(process.env.LD_SDK_KEY\!);
await client.waitForInitialization();

async function getFlag<T>(key: string, user: ld.LDUser, fallback: T): Promise<T> {
  return client.variation(key, user, fallback) as Promise<T>;
}

// Usage with context
const user: ld.LDUser = {
  key: userId,
  email: userEmail,
  custom: { plan: "pro", signupDate: "2025-01-15" },
};

const showNewCheckout = await getFlag("new-checkout-flow", user, false);
```

### Unleash (Self-Hosted)

```typescript
import { initialize, isEnabled } from "unleash-client";

const unleash = initialize({
  url: "https://unleash.internal.company.com/api",
  appName: "web-app",
  customHeaders: { Authorization: process.env.UNLEASH_API_KEY\! },
});

// Percentage rollout with stickiness
function isFeatureEnabled(flag: string, userId: string): boolean {
  return isEnabled(flag, { userId, sessionId: userId });
}
```

### Custom Flag System (Minimal)

```typescript
interface FeatureFlag {
  key: string;
  enabled: boolean;
  rolloutPercent: number; // 0-100
  allowList: string[];    // user IDs with forced access
}

function evaluateFlag(flag: FeatureFlag, userId: string): boolean {
  if (\!flag.enabled) return false;
  if (flag.allowList.includes(userId)) return true;
  // Deterministic hash for consistent assignment
  const hash = murmurHash3(flag.key + userId) % 100;
  return hash < flag.rolloutPercent;
}

// Murmur3 gives uniform distribution; don't use Math.random()
function murmurHash3(input: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return Math.abs(h);
}
```

## Gradual Rollout Strategy

### Phased Rollout Pattern

```
Phase 1: Internal team (allowList)          -> 0% rollout, team IDs only
Phase 2: Beta cohort                        -> 0% rollout, beta user IDs
Phase 3: 5% canary                          -> 5% rollout, monitor errors
Phase 4: 25% early majority                 -> 25% rollout, monitor metrics
Phase 5: 50% broad rollout                  -> 50% rollout, A/B comparison
Phase 6: 100% general availability          -> 100% rollout
Phase 7: Remove flag, delete dead code path -> Flag archived
```

### Kill Switch Pattern

```typescript
// Wrap risky features with circuit-breaker flags
async function processPayment(order: Order) {
  const useNewProcessor = await getFlag("new-payment-processor", order.user, false);

  if (useNewProcessor) {
    try {
      return await newPaymentProcessor.charge(order);
    } catch (error) {
      // Auto-disable on repeated failures
      await reportFlagIncident("new-payment-processor", error);
      return await legacyPaymentProcessor.charge(order);
    }
  }
  return await legacyPaymentProcessor.charge(order);
}
```

## A/B Test Design

### Assignment and Bucketing

```typescript
interface Experiment {
  key: string;
  variants: { id: string; weight: number }[];
  salt: string; // Unique per experiment to decorrelate
}

function assignVariant(experiment: Experiment, userId: string): string {
  const hash = murmurHash3(experiment.salt + userId) % 10000;
  let cumulative = 0;
  for (const variant of experiment.variants) {
    cumulative += variant.weight * 100; // weights are 0-100
    if (hash < cumulative) return variant.id;
  }
  return experiment.variants[0].id; // fallback to control
}

// Usage
const checkoutExperiment: Experiment = {
  key: "checkout-redesign-q1",
  variants: [
    { id: "control", weight: 50 },
    { id: "single-page", weight: 50 },
  ],
  salt: "checkout-redesign-q1-v1",
};

const variant = assignVariant(checkoutExperiment, userId);
analytics.track("experiment_assigned", {
  experiment: checkoutExperiment.key,
  variant,
  userId,
});
```

### Statistical Significance Calculation

```python
import numpy as np
from scipy import stats

def check_significance(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
    alpha: float = 0.05,
) -> dict:
    """Two-proportion z-test for A/B experiment."""
    p_control = control_conversions / control_total
    p_treatment = treatment_conversions / treatment_total
    p_pooled = (control_conversions + treatment_conversions) / (control_total + treatment_total)

    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
    z_stat = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        "control_rate": round(p_control, 4),
        "treatment_rate": round(p_treatment, 4),
        "lift": round((p_treatment - p_control) / p_control * 100, 2),
        "p_value": round(p_value, 4),
        "significant": p_value < alpha,
        "sample_size_adequate": min(control_total, treatment_total) >= 1000,
    }

# Example: 3.2% vs 3.8% conversion
result = check_significance(320, 10000, 380, 10000)
# {"control_rate": 0.032, "treatment_rate": 0.038, "lift": 18.75,
#  "p_value": 0.0234, "significant": True, "sample_size_adequate": True}
```

### Sample Size Calculator

```python
def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,  # relative, e.g., 0.05 = 5% lift
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Per-variant sample size needed."""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha * np.sqrt(2 * p1 * (1-p1)) +
          z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2))) / (p2 - p1)) ** 2
    return int(np.ceil(n))

# 3% baseline, want to detect 10% relative lift -> need ~35K per variant
required_sample_size(0.03, 0.10)  # 34,742
```

## Gotchas and Anti-Patterns

### Flag Debt Accumulation
- **Problem**: Flags stay in code long after rollout completes. Codebases accumulate hundreds of stale flags.
- **Fix**: Set expiration dates at creation. Add lint rules that flag (pun intended) flags older than 90 days. Track flag lifecycle in a spreadsheet or flag provider dashboard. Schedule monthly flag cleanup sprints.
### Sample Ratio Mismatch (SRM)
- **Problem**: 50/50 split shows 51.2/48.8 actual distribution. Assignment logic has a bug, or redirects/bot filtering disproportionately affect one variant.
- **Fix**: Run SRM checks daily using chi-squared test. If p < 0.001, halt the experiment and investigate. Never trust results from an experiment with SRM.
### The Peeking Problem
- **Problem**: Checking results daily and stopping when p < 0.05. This inflates false positive rate to 20-30%.
- **Fix**: Pre-register experiment duration based on sample size calculation. Use sequential testing (e.g., always-valid p-values) if you must monitor continuously. Never call a winner early without sequential correction.
