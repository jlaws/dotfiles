---
name: j-tdd
description: "Autonomous TDD implementation loop — write tests first, run to confirm failure, implement until green, lint, commit. Use when adding a feature or fixing a bug with test-driven development. Do NOT use for debugging existing failures (use /j-debug)."
argument-hint: "<feature or bug description>"
---

Feature/bug: $ARGUMENTS

If no arguments provided, ask what the user wants to implement.

## Execution Loop

Follow this strict sequence — do not skip or reorder steps:

1. **Explore**: Read existing code around the feature area. Identify edge cases, existing patterns, and test infrastructure (test runner, fixtures, helpers).

2. **Write tests**: Write comprehensive tests covering happy path, error cases, and edge cases. These tests MUST fail initially — if any pass, you're testing existing behavior (fix the test).

3. **Verify RED**: Run the full test suite. Confirm every new test fails for the expected reason (feature missing, not typos or import errors). Paste output.

4. **Implement**: Make each test pass one-by-one. After each change, run the full test suite. Write minimal code — don't add features beyond what the tests require.

5. **Lint**: Run the project's formatter and linter (check Makefile, package.json scripts, or pyproject.toml for commands). Fix all issues.

6. **Final verification**: Run the full test suite one last time. Paste output confirming all tests pass and output is clean.

7. **Commit**: Create a commit summarizing what was implemented and test coverage achieved.

Do not move to step 4 until ALL tests from step 2 are written.
Do not skip running tests between implementations in step 4.

---

## Test-Driven Development (TDD) Methodology

### Overview

Write the test first. Watch it fail. Write minimal code to pass.

**Core principle:** If you didn't watch the test fail, you don't know if it tests the right thing.

**Violating the letter of the rules is violating the spirit of the rules.**

### When to Use

**Always:**
- New features
- Bug fixes
- Refactoring
- Behavior changes

**Exceptions (ask your human partner):**
- Throwaway prototypes (but delete and TDD it properly after the spike)
- Generated code
- Configuration files

Thinking "skip TDD just this once"? Stop. That's rationalization.

### The Iron Law

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

Write code before the test? Delete it. Start over.

**No exceptions:**
- Don't keep it as "reference"
- Don't "adapt" it while writing tests
- Don't look at it
- Delete means delete

Implement fresh from tests. Period.

### Red-Green-Refactor

#### RED - Write Failing Test

One minimal test showing what should happen.

```typescript
test('retries failed operations 3 times', async () => {
  let attempts = 0;
  const operation = () => {
    attempts++;
    if (attempts < 3) throw new Error('fail');
    return 'success';
  };

  const result = await retryOperation(operation);

  expect(result).toBe('success');
  expect(attempts).toBe(3);
});
```

Requirements: one behavior, clear name, real code (no mocks unless unavoidable).

#### Verify RED - Watch It Fail

**MANDATORY. Never skip.**

```bash
npm test path/to/test.test.ts
```

Confirm:
- Test fails (not errors)
- Failure message is expected
- Fails because feature missing (not typos)

**Test passes?** You're testing existing behavior. Fix test.
**Test errors?** Fix error, re-run until it fails correctly.

#### GREEN - Minimal Code

Write simplest code to pass the test. Nothing more.

```typescript
async function retryOperation<T>(fn: () => Promise<T>): Promise<T> {
  for (let i = 0; i < 3; i++) {
    try {
      return await fn();
    } catch (e) {
      if (i === 2) throw e;
    }
  }
  throw new Error('unreachable');
}
```

Don't add features, refactor other code, or "improve" beyond the test.

#### Verify GREEN - Watch It Pass

**MANDATORY.**

```bash
npm test path/to/test.test.ts
```

Confirm: test passes, other tests still pass, output pristine (no errors/warnings).

**Test fails?** Fix code, not test. **Other tests fail?** Fix now.

#### REFACTOR - Clean Up

After green only: remove duplication, improve names, extract helpers.
Keep tests green. Don't add behavior.

Then: next failing test for next behavior.

### Good Tests

| Quality | Good | Bad |
|---------|------|-----|
| **Minimal** | One thing. "and" in name? Split it. | `test('validates email and domain and whitespace')` |
| **Clear** | Name describes behavior | `test('test1')` |
| **Shows intent** | Demonstrates desired API | Obscures what code should do |
| **Tests logic** | Exercises a decision, transformation, or path | `expect(config.timeout).toBe(5000)` -- restates source code |
| **Targets production** | Tests real application code | Tests for test helpers, factories, or fixtures |

### Common Rationalizations

| Excuse | Reality |
|--------|---------|
| "I'll write tests after" | Tests passing immediately prove nothing. You never saw them catch the bug. |
| "Keep as reference, write tests first" | You'll adapt it. That's testing after with extra steps. Delete means delete. |
| "Deleting X hours of work is wasteful" | Sunk cost fallacy. Keeping unverified code is technical debt you'll pay interest on. |
| "TDD is dogmatic, I'm being pragmatic" | TDD IS pragmatic. "Pragmatic" shortcuts = debugging in production = slower. |
| "Tests after achieve the same goals" | Tests-after = "what does this do?" Tests-first = "what should this do?" You test what you built, not what's required. |

### Red Flags - STOP and Start Over

- Code before test
- Test after implementation
- Test passes immediately
- Can't explain why test failed
- Tests added "later"
- Rationalizing "just this once"
- "I already manually tested it"
- "Tests after achieve the same purpose"
- "It's about spirit not ritual"
- "Keep as reference" or "adapt existing code"
- "Already spent X hours, deleting is wasteful"
- "TDD is dogmatic, I'm being pragmatic"
- "This is different because..."

**All of these mean: Delete code. Start over with TDD.**

### Verification Checklist

Before marking work complete:

- [ ] Every new function/method has a test
- [ ] Watched each test fail before implementing
- [ ] Each test failed for expected reason (feature missing, not typo)
- [ ] Wrote minimal code to pass each test
- [ ] All tests pass
- [ ] Output pristine (no errors, warnings)
- [ ] Tests use real code (mocks only if unavoidable)
- [ ] Edge cases and errors covered
- [ ] Every test exercises a decision point, transformation, or behavior path (not config/settings)
- [ ] All tests target production code (no tests for test helpers/fixtures/utilities)

Can't check all boxes? You skipped TDD. Start over.

### When Stuck

| Problem | Solution |
|---------|----------|
| Don't know how to test | Write wished-for API. Write assertion first. Ask your human partner. |
| Test too complicated | Design too complicated. Simplify interface. |
| Must mock everything | Code too coupled. Use dependency injection. |
| Test setup huge | Extract helpers. Still complex? Simplify design. |

### Debugging Integration

Bug found? Write failing test reproducing it. Follow TDD cycle. Test proves fix and prevents regression. Never fix bugs without a test.

### Final Rule

```
Production code -> test exists and failed first
Otherwise -> not TDD
```

No exceptions without your human partner's permission.

### Testing Anti-Patterns

**Core principle:** Test what the code does, not what the mocks do.

**The Iron Laws:**
1. NEVER test mock behavior
2. NEVER add test-only methods to production classes
3. NEVER mock without understanding dependencies

#### Anti-Pattern 1: Testing Mock Behavior

```typescript
// BAD: Testing that the mock exists
test('renders sidebar', () => {
  render(<Page />);
  expect(screen.getByTestId('sidebar-mock')).toBeInTheDocument();
});

// GOOD: Test real component or don't mock it
test('renders sidebar', () => {
  render(<Page />);  // Don't mock sidebar
  expect(screen.getByRole('navigation')).toBeInTheDocument();
});
```

#### Anti-Pattern 2: Test-Only Methods in Production

```typescript
// BAD: destroy() only used in tests
class Session {
  async destroy() { /* ... */ }
}

// GOOD: Test utilities handle test cleanup
export async function cleanupSession(session: Session) { /* ... */ }
```

#### Anti-Pattern 3: Mocking Without Understanding

Mock at the correct level — mock the slow part, preserve behavior the test needs.

```
BEFORE mocking any method:
  1. Ask: "What side effects does the real method have?"
  2. Ask: "Does this test depend on any of those side effects?"
  3. Ask: "Do I fully understand what this test needs?"
```

#### Anti-Pattern 4: Incomplete Mocks

Mock the COMPLETE data structure as it exists in reality, not just fields your immediate test uses.

#### Anti-Pattern 5: Integration Tests as Afterthought

Testing is part of implementation, not optional follow-up. TDD prevents this.

#### Anti-Pattern 6: Testing Configuration Instead of Logic

```typescript
// BAD: Restates source code
test('has correct default timeout', () => {
  expect(config.timeout).toBe(5000);
});

// GOOD: Test the logic that uses the config
test('retries up to configured max attempts', () => {
  const config = new AppConfig({ maxRetries: 3 });
  const service = new ApiService(config);
  mockApi.failTimes(2);
  const result = await service.fetchWithRetry('/data');
  expect(result.status).toBe(200);
});
```

#### Anti-Pattern 7: Testing Test Code

Don't write tests for test helpers, factories, or fixtures. They are validated implicitly when production tests that use them pass.

#### Quick Reference

| Anti-Pattern | Fix |
|--------------|-----|
| Assert on mock elements | Test real component or unmock it |
| Test-only methods in production | Move to test utilities |
| Mock without understanding | Understand dependencies first, mock minimally |
| Incomplete mocks | Mirror real API completely |
| Tests as afterthought | TDD - tests first |
| Config/settings assertions | Test the logic that uses the config |
| Tests for test code | Test production code; helpers are validated implicitly |
| Over-complex mocks | Consider integration tests |

**Mocks are tools to isolate, not things to test.**

---

## Code Quality

### Principles

| Principle | Rule |
|-----------|------|
| SRP | One reason to change per function/class |
| DRY | Extract after 2+ duplicates, not before |
| YAGNI | Solve today's problem, not tomorrow's hypothetical |
| Composition > Inheritance | Prefer protocols/interfaces |
| Explicit > Implicit | Clarity beats cleverness |
| Favor Uniformity | One way to do each thing |
| Follow Ecosystem Patterns | Go all-in on chosen framework's philosophy and idioms |
| External Configuration | Enable external config for components; follow ecosystem patterns |

### Code Smells Checklist

**Naming**
- Booleans: `is`/`has`/`can`/`should` prefix
- Functions: verb prefix (`get`, `create`, `handle`, `fetch`)
- Descriptive names; avoid abbreviations unless obvious

**Functions**
- Single responsibility, <30 lines
- Max 3 parameters; use parameter object beyond that
- Minimize side effects
- Extract complex conditionals into named functions

**Complexity**
- Max 2 levels nesting; use early returns
- Replace conditional chains with lookup maps/polymorphism

**Make Invalid States Unrepresentable**
- Use generics / type hints to catch issues at compile-time / static analysis
- Use specialized types where invalid inputs are unrepresentable
- No `any` in TypeScript (use `unknown`); no force unwraps in Swift (unless provably safe)
- Use `Optional` / `Option` for null safety
- Validate early at boundaries, convert to constrained types, pass constrained types downstream
- Priority: **compile-time > static analysis > runtime** for catching errors

### Anti-Patterns

**Code**
- **Premature abstraction** -- wait for 2+ concrete implementations
- **God objects** -- split by responsibility
- **Magic values** -- use named constants
- **Swallowed exceptions** -- handle meaningfully or propagate
- **Commented-out code** -- delete it, git has history

**Process**
- **Large PRs** -- keep small and focused
- **Skipping tests** -- costs more later
- **Vague commits** -- use `fix: prevent null pointer in user lookup`
- **TODOs without context** -- include why, when, ticket

### Style Defaults

| Rule | Value |
|------|-------|
| Indentation | 2 spaces (no tabs) |
| Line endings | LF (Unix) |
| Final newline | Always |
| Line length | 80-100 soft limit |
| File size | Under 300 lines |
| Test location | Colocated or parallel |

**Naming conventions:** JS/TS/Swift = `camelCase`, Python/Rust/Go = `snake_case`, Types = `PascalCase`, Constants = `SCREAMING_SNAKE_CASE`

**Import order** (separated by blank lines): 1. Standard library, 2. Third-party, 3. Local modules

### Lint Priority Triage

| Priority | Examples | When to Fix |
|----------|----------|-------------|
| High | Type errors blocking build, security vulns, runtime errors | Immediately |
| Medium | Missing type annotations, unused vars, style violations | Before commit |
| Low | Formatting inconsistencies, comment improvements | When convenient |

### Refactoring Decision Framework

- **Early returns** over nested conditionals
- **Parameter objects** when >3 params
- **Lookup maps** over conditional chains
- **Extract function** when a block needs a comment to explain intent
- **Typed errors** over generic catch-all

### Performance (Profile First)

**React/Next.js**: `React.memo`, `useMemo`, code splitting, virtual scrolling
**Database**: Index frequently queried fields, batch queries (N+1), pagination
**API**: SWR/React Query caching, debounce/throttle, parallel requests
**Bundle**: Tree-shake, dynamic imports, route-level code splitting
