---
name: technical-debt-remediation
description: "Use when analyzing, quantifying, or remediating technical debt. Provides debt inventory taxonomy, impact assessment frameworks, prioritized remediation patterns, and legacy modernization strategies."
---

# Technical Debt Remediation

## Debt Inventory

### Code Debt
- **Duplicated Code**: Exact duplicates, similar logic, repeated rules
- **Complex Code**: Cyclomatic complexity >10, nesting >3 levels, methods >50 lines, god classes >500 lines
- **Poor Structure**: Circular dependencies, feature envy, shotgun surgery

### Architecture Debt
- Missing/leaky abstractions, violated boundaries, monolithic components
- Outdated frameworks, deprecated APIs, unsupported dependencies

### Testing Debt
- Coverage gaps, missing integration/performance tests
- Brittle/flaky/slow tests

### Infrastructure Debt
- Manual deployment, no rollback, missing monitoring

## Impact Assessment

```
Debt Item: Duplicate user validation logic (5 files)
Time Impact: 2 hrs/bug fix, 4 hrs/feature change
Monthly: ~20 hours | Annual: 240 hrs x $150/hr = $36,000
```

| Risk Level | Criteria |
|------------|----------|
| Critical | Security vulnerabilities, data loss risk |
| High | Performance degradation, frequent outages |
| Medium | Developer frustration, slow delivery |
| Low | Code style, minor inefficiencies |

## Metrics Dashboard

```yaml
cyclomatic_complexity: { current: 15.2, target: 10.0 }
code_duplication: { current: 23%, target: 5% }
test_coverage: { unit: 45%, integration: 12%, target: 80%/60% }
dependency_health: { outdated_major: 12, security_vulns: 7 }
```

## Prioritized Remediation

### Quick Wins (Week 1-2)
```
1. Extract duplicate validation -> shared module (8 hrs, saves 20 hrs/mo)
2. Add error monitoring to payment service (4 hrs, saves 15 hrs/mo)
3. Automate deployment script (12 hrs, saves 40 hrs/mo)
```

### Medium-Term (Month 1-3)
```
1. Refactor God class -> 4 focused services (60 hrs, saves 30 hrs/mo)
2. Framework upgrade (80 hrs, +30% performance)
```

### Long-Term (Quarter 2-4)
```
1. Domain-Driven Design (200 hrs, -50% coupling)
2. Comprehensive test suite to 80/60/30% coverage (300 hrs, -70% bugs)
```

## Strangler Fig Pattern

```python
# Phase 1: Facade over legacy
class PaymentFacade:
    def process_payment(self, order):
        return self.legacy_processor.doPayment(order.to_legacy())

# Phase 2: New service alongside
class PaymentService:
    def process_payment(self, order): ...

# Phase 3: Feature-flagged migration
class PaymentFacade:
    def process_payment(self, order):
        if feature_flag("use_new_payment"):
            return self.new_service.process_payment(order)
        return self.legacy.doPayment(order.to_legacy())
```

**Progressive Rollout:** 5% -> 25% -> 50% -> 100% (24h observation between increases)
**Rollback triggers:** Error rate >1%, latency >2x baseline

## Prevention

```yaml
pre_commit_hooks:
  - complexity_check: "max 10"
  - duplication_check: "max 5%"
  - test_coverage: "min 80% for new code"

ci_pipeline:
  - dependency_audit: "no high vulnerabilities"
  - performance_test: "no regression >10%"
  - architecture_check: "no new violations"
```

## Forcing Functions

Make it hard to ignore problems by building discovery and enforcement into the workflow:

- **Canonical run scripts**: Provide scripts for running services locally (not just documentation). If setup is broken, someone finds out immediately
- **Encode standards in tooling**: Implement coding styles/principles in linters, formatters, pre-commit hooks, and coding agent prompts -- not just wikis
- **Tickets over TODOs**: File tickets with deadlines rather than adding `// TODO` comments that rot. TODOs without ticket references are invisible debt
- **Continuous releases**: Release services continuously. If deployment is painful, that pain surfaces immediately and gets fixed
- Subset of "fail fast, fail loud" -- discover failures early, make them hard to ignore, which naturally drives continuous upkeep

## Boy Scout Rule

Leave the code a little better than you found it.

- When encountering tech debt while working on a feature, **default towards fixing it** rather than working around it
- In PR reviews, ask others to consider taking care of nearby technical debt
- Don't require perfection -- use judgment weighing feature delay vs system improvement, but **default to making things a little better**
- Why: amortizes refactoring cost, builds a culture of quality, cheaper to fix when context is fresh
- **Distinction from scope creep**: small, proportional nearby improvements (rename a confusing variable, extract a helper, fix a broken docstring) are not the same as "while I'm in here..." full refactors. See `workflow:refactoring-patterns` for the boundary

## Stakeholder Summary Template

```markdown
## Executive Summary
- Current debt score: [X] (High)
- Monthly velocity loss: [X]%
- Recommended investment: [X] hours
- Expected ROI: [X]% over 12 months

## Key Risks
1. [Critical risk with impact]

## Proposed Actions
1. Immediate: [this week]
2. Short-term: [1 month]
3. Long-term: [6 months]
```

## Agent Team Mode

For comprehensive debt audits of large codebases where code, architecture, testing, and infrastructure debt can be analyzed independently.

### Team Configuration

```yaml
team:
  recommended_size: 4
  agent_roles:
    - name: code-debt-analyst
      type: Explore
      focus: "Duplication detection, complexity analysis, code smell inventory"
      skills_loaded: ["workflow:technical-debt-remediation", "workflow:code-quality"]
    - name: arch-debt-analyst
      type: Explore
      focus: "Boundary violations, dependency analysis, pattern drift"
      skills_loaded: ["workflow:technical-debt-remediation", "architecture:architecture-patterns"]
    - name: test-debt-analyst
      type: Explore
      focus: "Coverage gaps, flaky tests, missing integration tests"
      skills_loaded: ["workflow:technical-debt-remediation", "testing:language-testing-patterns"]
    - name: infra-debt-analyst
      type: Explore
      focus: "Deployment gaps, monitoring holes, dependency health"
      skills_loaded: ["workflow:technical-debt-remediation", "devops:observability"]
  file_ownership: "shared-read-only"
  lead_mode: "hands-on"
```

### Team Workflow

1. Lead defines scope and distributes debt inventory categories to analysts
2. All 4 analysts work in parallel on their domain
3. Each analyst produces: inventory items, impact scores (time cost, risk level), recommended remediation
4. Lead merges into unified metrics dashboard, prioritizes remediation plan
5. Lead produces Stakeholder Summary with cross-cutting insights

### Single-Agent Fallback

Without team mode, execute all phases sequentially (default behavior). Team mode is an optional enhancement.
