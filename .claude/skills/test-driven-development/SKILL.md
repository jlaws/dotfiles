---
name: test-driven-development
description: "Use when implementing changes with tests written first."
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Test-Driven Development

Write the test first. Watch it fail. Write the minimal code that passes.

**Why the order matters:** a test you never watched fail has not been shown to test anything. It may
assert on the wrong path, be mocked away, or restate the implementation. Watching it fail for the
expected reason is what proves it has teeth.

## When it applies

Default to TDD for new features, bug fixes, and behavior changes — anywhere you are deciding what the
code should do.

It buys less where there is no behavior to specify: generated code, configuration, and throwaway
spikes. A pure rename or mechanical refactor is already covered by the existing tests; the useful
question there is whether they still pass, not whether to write new ones.

## Red-Green-Refactor

### RED: write one failing test

One behavior, a name that describes it, real code rather than mocks where possible.

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

Run it, and read the failure. It should fail — not error — and fail because the behavior is missing,
not because of a typo. Two failure modes to recognize:

- **It passes.** You are describing behavior that already exists. The test is not about your change.
- **It errors.** Fix the error and re-run until it fails for the right reason.

### GREEN: make it pass, and stop

Write the simplest code that satisfies the test, complete in one pass rather than adding pieces and
re-running. Do not add features, refactor neighbors, or improve beyond what the test asks.

Then confirm the new test passes, the rest still pass, and the output is clean. If the new test still
fails, fix the code rather than the test.

**Passing tests mean stop.** Do not polish code that already passes unless asked.

### REFACTOR: only once green

Remove duplication, improve names, extract helpers. Keep tests green and add no behavior. Then write
the next failing test.

## What makes a test worth having

| Quality | Good | Bad |
|---------|------|-----|
| **Minimal** | One thing. An "and" in the name means split it. | `test('validates email and domain and whitespace')` |
| **Clear** | The name describes the behavior | `test('test1')` |
| **Shows intent** | Demonstrates the API you want | Obscures what the code should do |
| **Tests logic** | Exercises a decision, transformation, or path | `expect(config.timeout).toBe(5000)` — restates the source |
| **Targets production** | Tests real application code | Tests the helpers, factories, or fixtures |

## When stuck

| Problem | What it usually means |
|---------|----------------------|
| Don't know how to test it | Write the API you wish existed, then the assertion. Ask if still unclear. |
| Test is complicated | The design is complicated. Simplify the interface. |
| Must mock everything | The code is too coupled. Inject dependencies. |
| Setup is huge | Extract helpers; if it stays complex, simplify the design. |
| Same failure twice | Stop after two attempts and rethink the approach. |

## Bugs

Reproduce the bug as a failing test before fixing it, so the fix is proven and the bug cannot come
back silently. Name it for the scenario (`test_regression_null_user_concurrent_login`), fix minimally,
then add boundary variants around the fix.

**The test must fail without the fix and pass with it.** If you cannot show both states, it is not
testing the thing you think it is.

## Related

If you are unsure whether a mocking approach is sound — testing mock behavior instead of real
behavior, test-only methods on production classes, tautology tests that assert configuration — read
`references/testing-anti-patterns.md`.

A shipped change also needs a documentation decision: updated, or N/A with a reason. See
`documentation-validation`.
