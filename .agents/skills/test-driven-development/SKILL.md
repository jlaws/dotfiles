---
name: test-driven-development
description: Use when implementing features, fixing bugs, or refactoring — enforces writing tests before implementation code. Do NOT use for investigating existing bugs or failures (use debugging-methodology) or pre-commit verification (use verification-before-completion).
---

# Test-Driven Development (TDD)

## Overview

Write the test first. Watch it fail. Write minimal code to pass.

**Core principle:** If you didn't watch the test fail, you don't know if it tests the right thing.

**Violating the letter of the rules is violating the spirit of the rules.**

## When to Use

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

## The Iron Law

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

## Red-Green-Refactor

### RED - Write Failing Test

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

### Verify RED - Watch It Fail

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

### GREEN - Minimal Code

Write simplest code to pass the test. Nothing more. Write the complete solution in one pass — not incrementally (partial code → test → add more → test again wastes cycles).

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

**Stop when green.** Once tests pass, you are done. Do not refactor, polish, or improve passing code unless explicitly asked. Passing tests = stop.

### Verify GREEN - Watch It Pass

**MANDATORY.**

```bash
npm test path/to/test.test.ts
```

Confirm: test passes, other tests still pass, output pristine (no errors/warnings).

**Test fails?** Fix code, not test. **Other tests fail?** Fix now.

### REFACTOR - Clean Up

After green only: remove duplication, improve names, extract helpers.
Keep tests green. Don't add behavior.

Then: next failing test for next behavior.

## Good Tests

| Quality | Good | Bad |
|---------|------|-----|
| **Minimal** | One thing. "and" in name? Split it. | `test('validates email and domain and whitespace')` |
| **Clear** | Name describes behavior | `test('test1')` |
| **Shows intent** | Demonstrates desired API | Obscures what code should do |
| **Tests logic** | Exercises a decision, transformation, or path | `expect(config.timeout).toBe(5000)` -- restates source code |
| **Targets production** | Tests real application code | Tests for test helpers, factories, or fixtures |

## Common Rationalizations

| Excuse | Reality |
|--------|---------|
| "I'll write tests after" | Tests passing immediately prove nothing. You never saw them catch the bug. |
| "Keep as reference, write tests first" | You'll adapt it. That's testing after with extra steps. Delete means delete. |
| "Deleting X hours of work is wasteful" | Sunk cost fallacy. Keeping unverified code is technical debt you'll pay interest on. |
| "TDD is dogmatic, I'm being pragmatic" | TDD IS pragmatic. "Pragmatic" shortcuts = debugging in production = slower. |
| "Tests after achieve the same goals" | Tests-after = "what does this do?" Tests-first = "what should this do?" You test what you built, not what's required. |

## Red Flags - STOP and Start Over

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

## Verification Checklist

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

## When Stuck

| Problem | Solution |
|---------|----------|
| Don't know how to test | Write wished-for API. Write assertion first. Ask your human partner. |
| Test too complicated | Design too complicated. Simplify interface. |
| Must mock everything | Code too coupled. Use dependency injection. |
| Test setup huge | Extract helpers. Still complex? Simplify design. |
| Same error twice | Max 2 fix attempts on the same failure. If still failing, rethink the approach entirely. |

## Debugging Integration

Bug found? Write failing test reproducing it. Follow TDD cycle. Test proves fix and prevents regression. Never fix bugs without a test.

### Regression Test Pattern

Formalized workflow for bug-driven test creation:

1. **Reproduce** — Write a failing test with the exact scenario that triggered the bug
2. **Name it** — `test_regression_{description}_{scenario}` (e.g., `test_regression_null_user_concurrent_login`)
3. **Fix** — Minimal code change to make the test pass
4. **Harden** — Add boundary variants around the fix

| Bug Type | Regression Test Pattern |
|----------|----------------------|
| Null/undefined crash | Test with null, empty, and boundary inputs |
| Off-by-one | Test at boundary, boundary-1, boundary+1 |
| Race condition | Test concurrent access with shared state |
| State corruption | Test state transitions in exact failing sequence |
| Parsing failure | Test with exact malformed input + similar variants |
| Auth bypass | Test with each role/permission that should be denied |

**Key rule:** The regression test must fail without the fix and pass with it. If you can't demonstrate both states, you're not testing the right thing.

## Testing Anti-Patterns

If you are unsure whether your mock approach introduces a known anti-pattern, read `references/testing-anti-patterns.md`:
- Testing mock behavior instead of real behavior
- Adding test-only methods to production classes
- Mocking without understanding dependencies
- Testing configuration instead of logic (tautology tests)

## Final Rule

```
Production code -> test exists and failed first
Otherwise -> not TDD
```

No exceptions without your human partner's permission.
