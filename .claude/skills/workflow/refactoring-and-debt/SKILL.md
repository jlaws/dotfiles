---
name: refactoring-and-debt
description: "Systematic code refactoring and technical debt analysis with safe operations and test-verify-commit cadence. Use when systematically refactoring code or analyzing/remediating technical debt. Covers safe refactoring operations, test-verify-commit cadence, and debt inventory. Do NOT use for general code quality checks (use code-quality) or pre-commit verification (use verification-before-completion)."
compatibility: claude-code
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Refactoring & Technical Debt

## The Discipline

```
TEST -> REFACTOR -> VERIFY -> COMMIT
Never skip a step. Never combine refactoring with behavior changes.
```

A refactoring changes structure without changing behavior. If you're adding features or fixing bugs at the same time, you're not refactoring -- you're gambling.

### The Cadence

1. **Ensure tests pass** (run full suite, green baseline)
2. **Make ONE structural change** (single refactoring operation)
3. **Run tests again** (must still pass -- if not, revert immediately)
4. **Commit** (small atomic commit, message describes the refactoring)
5. **Repeat**

Each commit is a safe rollback point. If anything breaks, `git revert` is trivial.

## Code Smell to Refactoring Mapping

| Code Smell | Refactoring(s) |
|------------|-----------------|
| Long method (>30 lines) | Extract function |
| Long parameter list (>3) | Introduce parameter object |
| Duplicated code | Extract function / Extract base class |
| Feature envy (method uses another class's data) | Move method |
| Data clumps (same fields grouped repeatedly) | Extract class / Introduce parameter object |
| Switch on type in multiple places | Replace conditional with polymorphism |
| Divergent change (class changed for multiple reasons) | Extract class (split by responsibility) |
| Shotgun surgery (one change touches many files) | Move method / Inline class (consolidate) |
| Primitive obsession (strings/ints for domain concepts) | Introduce value object |
| Speculative generality (unused abstractions) | Inline class / Remove dead code |
| Dead code | Delete it. Git has history. |

## Safe Refactoring Sequences

| Goal | Sequence |
|------|----------|
| Break up god class | Extract methods -> Group related -> Extract classes -> Define interfaces |
| Remove inheritance | Push down unused methods -> Extract interface -> Replace with delegation -> Remove base |
| Simplify complex conditional | Extract each branch into named function -> Replace with lookup map or polymorphism |
| Migrate to new API | Introduce adapter -> Route calls through adapter -> Swap implementation -> Inline adapter |

## IDE-Assisted vs Manual

| Refactoring | IDE? | Notes |
|-------------|:---:|-------|
| Rename | Yes | Always use IDE -- catches all references |
| Extract function | Yes | IDE infers parameters and return type |
| Move file/module | Yes | Updates import paths automatically |
| Change signature | Yes | IDE updates all call sites |
| Replace conditional with polymorphism | No | Requires architectural judgment |
| Strangler fig / Branch by abstraction | No | Strategy, not mechanical transformation |

**Rule**: If your IDE can do it, let your IDE do it.

## Large-Scale Refactoring Strategies

### Strangler Fig

Gradually replace a legacy system by routing new functionality to a new implementation while keeping the old one running.

1. Identify the boundary (API, module interface, route)
2. Build new implementation behind the same interface
3. Route traffic/calls incrementally (feature flag, proxy, or router)
4. Monitor both paths (correctness + performance)
5. Remove old implementation when 100% migrated

**Progressive rollout:** 5% -> 25% -> 50% -> 100% (24h observation between increases)
**Rollback triggers:** Error rate >1%, latency >2x baseline

### Branch by Abstraction

1. Create an interface/protocol wrapping the current implementation
2. Update all callers to use the abstraction (test + commit)
3. Build new implementation behind the same abstraction
4. Switch (toggle, config, or swap) to new implementation
5. Remove old implementation and (optionally) the abstraction

### Parallel Implementation

Run old and new code simultaneously, compare outputs, converge when confident. Best for high-risk logic changes where correctness is critical (payments, data pipelines).

> See `references/refactoring-catalog.md` for detailed code examples of each refactoring operation.

## Technical Debt Inventory

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

| Risk Level | Criteria |
|------------|----------|
| Critical | Security vulnerabilities, data loss risk |
| High | Performance degradation, frequent outages |
| Medium | Developer frustration, slow delivery |
| Low | Code style, minor inefficiencies |

**Quantify**: `Debt Item -> hrs/incident x frequency = monthly cost x rate = annual cost`

## Prioritized Remediation

| Horizon | Examples | Criteria |
|---------|----------|----------|
| Quick wins (1-2 weeks) | Extract shared module, add monitoring, automate deploy | High savings/effort ratio |
| Medium-term (1-3 months) | Refactor god classes, framework upgrade | Moderate effort, measurable ROI |
| Long-term (2-4 quarters) | DDD migration, comprehensive test suite | Strategic investment |

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

- **Canonical run scripts**: Provide scripts for running services locally. If setup is broken, someone finds out immediately
- **Encode standards in tooling**: Linters, formatters, pre-commit hooks, coding agent prompts -- not just wikis
- **Tickets over TODOs**: File tickets with deadlines rather than adding `// TODO` comments that rot
- **Continuous releases**: If deployment is painful, that pain surfaces immediately and gets fixed

## Boy Scout Rule

Leave the code a little better than you found it.

- When encountering tech debt while working on a feature, **default towards fixing it** rather than working around it
- In PR reviews, ask others to consider taking care of nearby technical debt
- **Proportionality**: small nearby improvements (rename, extract helper, fix docstring) are encouraged; full refactors should be separate tickets

## When NOT to Refactor

| Situation | Why Not |
|-----------|---------|
| No tests covering the code | Can't verify behavior preservation. Write tests first. |
| Under deadline pressure | Ship first, refactor next sprint. |
| Code being deleted soon | Don't polish what you're throwing away. |
| "While I'm in here..." scope creep | File a ticket, do it separately. Exception: boy scout rule. |
| Single implementation | Don't create abstractions for one concrete case. Wait for the second. |

## Gotchas

- Refactoring without tests is walking a tightrope without a net. Write characterization tests first if coverage is low
- "Refactoring" that changes behavior is rewriting -- separate into distinct commits
- Large PRs labeled "refactoring" are suspicious; each step should be its own atomic green commit
- Branch by abstraction's "temporary" interface layer tends to become permanent; set a deadline

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

For comprehensive debt audits of large codebases.

```yaml
team:
  recommended_size: 4
  agent_roles:
    - name: code-debt-analyst
      type: Explore
      focus: "Duplication, complexity, code smell inventory"
      skills: ["workflow:refactoring-and-debt", "workflow:code-quality"]
    - name: arch-debt-analyst
      type: Explore
      focus: "Boundary violations, dependency analysis, pattern drift"
      skills: ["workflow:refactoring-and-debt"]
      references: [".claude/references/architecture/architecture-decision-records.md"]
    - name: test-debt-analyst
      type: Explore
      focus: "Coverage gaps, flaky tests, missing integration tests"
      skills: ["workflow:refactoring-and-debt", "testing:language-testing-patterns"]
    - name: infra-debt-analyst
      type: Explore
      focus: "Deployment gaps, monitoring holes, dependency health"
      skills: ["workflow:refactoring-and-debt"]
      references: [".claude/references/devops/observability.md"]
  file_ownership: "shared-read-only"
  lead_mode: "hands-on"
```

## Cross-References

- **workflow:code-quality** -- code smells, style conventions
- **workflow:verification-before-completion** -- ensuring refactoring is verified before claiming done
