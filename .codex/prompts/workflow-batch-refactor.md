---
name: workflow-batch-refactor
description: "Batch refactoring across many files. Use when refactoring a concept, pattern, or API across 10+ files. Do NOT use for small refactors touching <5 files (just do it directly) or behavior changes (use /brainstorm + /write-plan)."
argument-hint: "<refactoring description>"
---

Refactoring: $ARGUMENTS

If no arguments provided, ask what the user wants to refactor.

## Coordination Plan

Follow this sequence:

### Phase 1: Discovery
1. Identify EVERY file that needs changes (use Grep/Glob exhaustively)
2. Group files into independent batches (default: 5 batches). Files in the same batch must not depend on each other.
3. Present the batch plan to the user for approval before proceeding.

### Phase 2: Sequential Batch Execution
For each batch, follow the per-task execution flow (see `references/workflow/task-execution-checklists`):
1. Make the specified changes to the assigned files
2. Self-review: completeness, consistency, correctness, convention (see Per-Batch checklist below)
3. Spec compliance check: did I change every instance? Any files missed?
4. Code quality check: is the pattern applied consistently?
5. Run lint/format on changed files
6. Verify tests still pass
7. Commit changes with a descriptive message: `batch N: <summary>`

### Phase 3: Verification
After all batches complete:
1. Run full lint validation
2. Run tests if applicable
3. Review the cumulative diff for consistency

### Phase 4: PR
Use the create-pr workflow to open the PR with a summary of all changes.

> **Worktree isolation:** For batch refactoring that needs isolation from current work, see `skill:using-git-worktrees` for setup, safety verification, and completion steps.

### Per-Batch Execution Checklist

For each batch, follow this discipline:

#### Implementation
1. Read ALL files in the batch before making changes
2. Apply the refactoring pattern consistently across all files
3. Verify each file compiles/parses after changes

#### Self-Review Before Committing
- **Completeness**: Did I change every instance? Any files missed?
- **Consistency**: Is the pattern applied the same way in every file?
- **Correctness**: Do the changes preserve behavior? Any edge cases?
- **Convention**: Do changes follow existing codebase patterns?

#### Verification
1. Run lint/format on changed files
2. Run tests — if any fail, fix before proceeding
3. Review the diff for unintended changes

#### Red Flags — Stop and Reassess
- Batch requires changes to files outside its scope
- Refactoring changes behavior (not just structure)
- Test failures that aren't obviously related to the refactoring
- Merge conflicts between batches (batches weren't truly independent)

---

## Refactoring & Technical Debt

### The Discipline

```
TEST -> REFACTOR -> VERIFY -> COMMIT
Never skip a step. Never combine refactoring with behavior changes.
```

A refactoring changes structure without changing behavior. If you're adding features or fixing bugs at the same time, you're not refactoring -- you're gambling.

#### The Cadence

1. **Ensure tests pass** (run full suite, green baseline)
2. **Make ONE structural change** (single refactoring operation)
3. **Run tests again** (must still pass -- if not, revert immediately)
4. **Commit** (small atomic commit, message describes the refactoring)
5. **Repeat**

Each commit is a safe rollback point. If anything breaks, `git revert` is trivial.

### Code Smell to Refactoring Mapping

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

### Safe Refactoring Sequences

| Goal | Sequence |
|------|----------|
| Break up god class | Extract methods -> Group related -> Extract classes -> Define interfaces |
| Remove inheritance | Push down unused methods -> Extract interface -> Replace with delegation -> Remove base |
| Simplify complex conditional | Extract each branch into named function -> Replace with lookup map or polymorphism |
| Migrate to new API | Introduce adapter -> Route calls through adapter -> Swap implementation -> Inline adapter |

### IDE-Assisted vs Manual

| Refactoring | IDE? | Notes |
|-------------|:---:|-------|
| Rename | Yes | Always use IDE -- catches all references |
| Extract function | Yes | IDE infers parameters and return type |
| Move file/module | Yes | Updates import paths automatically |
| Change signature | Yes | IDE updates all call sites |
| Replace conditional with polymorphism | No | Requires architectural judgment |
| Strangler fig / Branch by abstraction | No | Strategy, not mechanical transformation |

**Rule**: If your IDE can do it, let your IDE do it.

### Large-Scale Refactoring Strategies

#### Strangler Fig

Gradually replace a legacy system by routing new functionality to a new implementation while keeping the old one running.

1. Identify the boundary (API, module interface, route)
2. Build new implementation behind the same interface
3. Route traffic/calls incrementally (feature flag, proxy, or router)
4. Monitor both paths (correctness + performance)
5. Remove old implementation when 100% migrated

**Progressive rollout:** 5% -> 25% -> 50% -> 100% (24h observation between increases)
**Rollback triggers:** Error rate >1%, latency >2x baseline

#### Branch by Abstraction

1. Create an interface/protocol wrapping the current implementation
2. Update all callers to use the abstraction (test + commit)
3. Build new implementation behind the same abstraction
4. Switch (toggle, config, or swap) to new implementation
5. Remove old implementation and (optionally) the abstraction

#### Parallel Implementation

Run old and new code simultaneously, compare outputs, converge when confident. Best for high-risk logic changes where correctness is critical (payments, data pipelines).

### Refactoring Catalog — Code Examples

#### Extract Function / Method

**When**: Code block needs a comment to explain intent, or is duplicated.

```python
# Before
def process_order(order):
    # Calculate discount
    if order.customer.is_premium and order.total > 100:
        discount = order.total * 0.15
    elif order.total > 200:
        discount = order.total * 0.10
    else:
        discount = 0
    order.total -= discount
    # ... more processing

# After
def calculate_discount(order):
    if order.customer.is_premium and order.total > 100:
        return order.total * 0.15
    if order.total > 200:
        return order.total * 0.10
    return 0

def process_order(order):
    order.total -= calculate_discount(order)
    # ... more processing
```

#### Extract Class / Module

**When**: A class has multiple responsibilities or a module is >300 lines with distinct sections.

Split by responsibility. Each resulting unit should have a single reason to change.

#### Inline Function / Variable

**When**: Indirection adds no clarity. The function body is as clear as the name.

```typescript
// Before
function isEligible(age: number): boolean {
    return age >= 18;
}
const eligible = isEligible(user.age);

// After (if only used once and meaning is obvious)
const eligible = user.age >= 18;
```

#### Replace Conditional with Polymorphism

**When**: Switch/if-else chain on a type field that appears in 3+ places.

```typescript
// Before
function getArea(shape: Shape): number {
    switch (shape.type) {
        case 'circle': return Math.PI * shape.radius ** 2;
        case 'rectangle': return shape.width * shape.height;
        case 'triangle': return 0.5 * shape.base * shape.height;
    }
}

// After
interface Shape {
    getArea(): number;
}
class Circle implements Shape {
    getArea() { return Math.PI * this.radius ** 2; }
}
class Rectangle implements Shape {
    getArea() { return this.width * this.height; }
}
```

#### Replace Inheritance with Composition

**When**: Subclass only uses a fraction of parent, or "is-a" relationship is forced.

```python
# Before
class AudioPlayer(MediaWidget):  # inherits 50 methods, uses 5
    pass

# After
class AudioPlayer:
    def __init__(self):
        self.media = MediaWidget()  # delegate what you need
```

#### Introduce Parameter Object

**When**: 3+ parameters travel together across multiple functions.

```go
// Before
func createUser(name string, email string, age int, role string, dept string) {}

// After
type CreateUserParams struct {
    Name  string
    Email string
    Age   int
    Role  string
    Dept  string
}
func createUser(params CreateUserParams) {}
```

#### Replace Magic Values with Constants

```python
# Before
if response.status_code == 429:
    time.sleep(60)

# After
RATE_LIMIT_STATUS = 429
RATE_LIMIT_COOLDOWN_SECONDS = 60
if response.status_code == RATE_LIMIT_STATUS:
    time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
```

#### Strangler Fig — Detailed Example

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

#### Parallel Implementation

```python
def process(data):
    old_result = old_implementation(data)
    new_result = new_implementation(data)
    if old_result != new_result:
        log.warning(f"Mismatch: {old_result} vs {new_result}")
    return old_result  # switch to new_result when confident
```

#### Metrics Dashboard Template

```yaml
cyclomatic_complexity: { current: 15.2, target: 10.0 }
code_duplication: { current: 23%, target: 5% }
test_coverage: { unit: 45%, integration: 12%, target: 80%/60% }
dependency_health: { outdated_major: 12, security_vulns: 7 }
```

#### Impact Assessment Example

```
Debt Item: Duplicate user validation logic (5 files)
Time Impact: 2 hrs/bug fix, 4 hrs/feature change
Monthly: ~20 hours | Annual: 240 hrs x $150/hr = $36,000
```

### Technical Debt Inventory

#### Code Debt
- **Duplicated Code**: Exact duplicates, similar logic, repeated rules
- **Complex Code**: Cyclomatic complexity >10, nesting >3 levels, methods >50 lines, god classes >500 lines
- **Poor Structure**: Circular dependencies, feature envy, shotgun surgery

#### Architecture Debt
- Missing/leaky abstractions, violated boundaries, monolithic components
- Outdated frameworks, deprecated APIs, unsupported dependencies

#### Testing Debt
- Coverage gaps, missing integration/performance tests
- Brittle/flaky/slow tests

#### Infrastructure Debt
- Manual deployment, no rollback, missing monitoring

### Impact Assessment

| Risk Level | Criteria |
|------------|----------|
| Critical | Security vulnerabilities, data loss risk |
| High | Performance degradation, frequent outages |
| Medium | Developer frustration, slow delivery |
| Low | Code style, minor inefficiencies |

**Quantify**: `Debt Item -> hrs/incident x frequency = monthly cost x rate = annual cost`

### Prioritized Remediation

| Horizon | Examples | Criteria |
|---------|----------|----------|
| Quick wins (1-2 weeks) | Extract shared module, add monitoring, automate deploy | High savings/effort ratio |
| Medium-term (1-3 months) | Refactor god classes, framework upgrade | Moderate effort, measurable ROI |
| Long-term (2-4 quarters) | DDD migration, comprehensive test suite | Strategic investment |

### Prevention

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

### Forcing Functions

- **Canonical run scripts**: Provide scripts for running services locally. If setup is broken, someone finds out immediately
- **Encode standards in tooling**: Linters, formatters, pre-commit hooks, coding agent prompts -- not just wikis
- **Tickets over TODOs**: File tickets with deadlines rather than adding `// TODO` comments that rot
- **Continuous releases**: If deployment is painful, that pain surfaces immediately and gets fixed

### Boy Scout Rule

Leave the code a little better than you found it.

- When encountering tech debt while working on a feature, **default towards fixing it** rather than working around it
- In PR reviews, ask others to consider taking care of nearby technical debt
- **Proportionality**: small nearby improvements (rename, extract helper, fix docstring) are encouraged; full refactors should be separate tickets

### When NOT to Refactor

| Situation | Why Not |
|-----------|---------|
| No tests covering the code | Can't verify behavior preservation. Write tests first. |
| Under deadline pressure | Ship first, refactor next sprint. |
| Code being deleted soon | Don't polish what you're throwing away. |
| "While I'm in here..." scope creep | File a ticket, do it separately. Exception: boy scout rule. |
| Single implementation | Don't create abstractions for one concrete case. Wait for the second. |

### Gotchas

- Refactoring without tests is walking a tightrope without a net. Write characterization tests first if coverage is low
- "Refactoring" that changes behavior is rewriting -- separate into distinct commits
- Large PRs labeled "refactoring" are suspicious; each step should be its own atomic green commit
- Branch by abstraction's "temporary" interface layer tends to become permanent; set a deadline

### Stakeholder Summary Template

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

### Cross-References

- **workflow:code-quality** -- code smells, style conventions
- **workflow:verification-before-completion** -- ensuring refactoring is verified before claiming done
