# Testing Anti-Patterns

**Load this reference when:** writing or changing tests, adding mocks, or tempted to add test-only methods to production code.

## Overview

Tests must verify real behavior, not mock behavior. Mocks are a means to isolate, not the thing being tested.

**Core principle:** Test what the code does, not what the mocks do.

**Following strict TDD prevents these anti-patterns.**

## The Iron Laws

```
1. NEVER test mock behavior
2. NEVER add test-only methods to production classes
3. NEVER mock without understanding dependencies
```

## Anti-Pattern 1: Testing Mock Behavior

**The violation:**
```typescript
// ❌ BAD: Testing that the mock exists
test('renders sidebar', () => {
  render(<Page />);
  expect(screen.getByTestId('sidebar-mock')).toBeInTheDocument();
});
```

**Why this is wrong:**
- You're verifying the mock works, not that the component works
- Test passes when mock is present, fails when it's not
- Tells you nothing about real behavior

**your human partner's correction:** "Are we testing the behavior of a mock?"

**The fix:**
```typescript
// ✅ GOOD: Test real component or don't mock it
test('renders sidebar', () => {
  render(<Page />);  // Don't mock sidebar
  expect(screen.getByRole('navigation')).toBeInTheDocument();
});

// OR if sidebar must be mocked for isolation:
// Don't assert on the mock - test Page's behavior with sidebar present
```

### Gate Function

```
BEFORE asserting on any mock element:
  Ask: "Am I testing real component behavior or just mock existence?"

  IF testing mock existence:
    STOP - Delete the assertion or unmock the component

  Test real behavior instead
```

## Anti-Pattern 2: Test-Only Methods in Production

**The violation:**
```typescript
// ❌ BAD: destroy() only used in tests
class Session {
  async destroy() {  // Looks like production API!
    await this._workspaceManager?.destroyWorkspace(this.id);
    // ... cleanup
  }
}

// In tests
afterEach(() => session.destroy());
```

**Why this is wrong:**
- Production class polluted with test-only code
- Dangerous if accidentally called in production
- Violates YAGNI and separation of concerns
- Confuses object lifecycle with entity lifecycle

**The fix:**
```typescript
// ✅ GOOD: Test utilities handle test cleanup
// Session has no destroy() - it's stateless in production

// In test-utils/
export async function cleanupSession(session: Session) {
  const workspace = session.getWorkspaceInfo();
  if (workspace) {
    await workspaceManager.destroyWorkspace(workspace.id);
  }
}

// In tests
afterEach(() => cleanupSession(session));
```

### Gate Function

```
BEFORE adding any method to production class:
  Ask: "Is this only used by tests?"

  IF yes:
    STOP - Don't add it
    Put it in test utilities instead

  Ask: "Does this class own this resource's lifecycle?"

  IF no:
    STOP - Wrong class for this method
```

## Anti-Pattern 3: Mocking Without Understanding

**The violation:**
```typescript
// ❌ BAD: Mock breaks test logic
test('detects duplicate server', () => {
  // Mock prevents config write that test depends on!
  vi.mock('ToolCatalog', () => ({
    discoverAndCacheTools: vi.fn().mockResolvedValue(undefined)
  }));

  await addServer(config);
  await addServer(config);  // Should throw - but won't!
});
```

**Why this is wrong:**
- Mocked method had side effect test depended on (writing config)
- Over-mocking to "be safe" breaks actual behavior
- Test passes for wrong reason or fails mysteriously

**The fix:**
```typescript
// ✅ GOOD: Mock at correct level
test('detects duplicate server', () => {
  // Mock the slow part, preserve behavior test needs
  vi.mock('MCPServerManager'); // Just mock slow server startup

  await addServer(config);  // Config written
  await addServer(config);  // Duplicate detected ✓
});
```

### Gate Function

```
BEFORE mocking any method:
  STOP - Don't mock yet

  1. Ask: "What side effects does the real method have?"
  2. Ask: "Does this test depend on any of those side effects?"
  3. Ask: "Do I fully understand what this test needs?"

  IF depends on side effects:
    Mock at lower level (the actual slow/external operation)
    OR use test doubles that preserve necessary behavior
    NOT the high-level method the test depends on

  IF unsure what test depends on:
    Run test with real implementation FIRST
    Observe what actually needs to happen
    THEN add minimal mocking at the right level

  Red flags:
    - "I'll mock this to be safe"
    - "This might be slow, better mock it"
    - Mocking without understanding the dependency chain
```

## Anti-Pattern 4: Incomplete Mocks

**The violation:**
```typescript
// ❌ BAD: Partial mock - only fields you think you need
const mockResponse = {
  status: 'success',
  data: { userId: '123', name: 'Alice' }
  // Missing: metadata that downstream code uses
};

// Later: breaks when code accesses response.metadata.requestId
```

**Why this is wrong:**
- **Partial mocks hide structural assumptions** - You only mocked fields you know about
- **Downstream code may depend on fields you didn't include** - Silent failures
- **Tests pass but integration fails** - Mock incomplete, real API complete
- **False confidence** - Test proves nothing about real behavior

**The Iron Rule:** Mock the COMPLETE data structure as it exists in reality, not just fields your immediate test uses.

**The fix:**
```typescript
// ✅ GOOD: Mirror real API completeness
const mockResponse = {
  status: 'success',
  data: { userId: '123', name: 'Alice' },
  metadata: { requestId: 'req-789', timestamp: 1234567890 }
  // All fields real API returns
};
```

### Gate Function

```
BEFORE creating mock responses:
  Check: "What fields does the real API response contain?"

  Actions:
    1. Examine actual API response from docs/examples
    2. Include ALL fields system might consume downstream
    3. Verify mock matches real response schema completely

  Critical:
    If you're creating a mock, you must understand the ENTIRE structure
    Partial mocks fail silently when code depends on omitted fields

  If uncertain: Include all documented fields
```

## Anti-Pattern 5: Integration Tests as Afterthought

**The violation:**
```
✅ Implementation complete
❌ No tests written
"Ready for testing"
```

**Why this is wrong:**
- Testing is part of implementation, not optional follow-up
- TDD would have caught this
- Can't claim complete without tests

**The fix:**
```
TDD cycle:
1. Write failing test
2. Implement to pass
3. Refactor
4. THEN claim complete
```

## Anti-Pattern 6: Testing Configuration Instead of Logic

**The violation:**
```typescript
// ❌ BAD: Testing that config defaults exist
test('has correct default timeout', () => {
  const config = new AppConfig();
  expect(config.timeout).toBe(5000);  // Restates source code
});

// ❌ BAD: Testing route registration
test('registers /users endpoint', () => {
  const routes = app.getRoutes();
  expect(routes).toContainEqual({ path: '/users', method: 'GET' });
});

// ❌ BAD: Testing enum values
test('Status has ACTIVE value', () => {
  expect(Status.ACTIVE).toBe('active');
});

// ❌ BAD: "Renders without crashing" with no assertion
test('renders without crashing', () => {
  render(<UserProfile />);
  // ...nothing checked
});
```

**Why this is wrong:**
- Tautologies that restate source code as assertions
- No decision, transformation, or behavior path exercised
- Pass by definition — can only fail if source code changes (which is what you want)
- Inflate coverage without catching any bugs

**The fix:**
```typescript
// ✅ GOOD: Test retry behavior driven by config
test('retries up to configured max attempts', () => {
  const config = new AppConfig({ maxRetries: 3 });
  const service = new ApiService(config);
  mockApi.failTimes(2);

  const result = await service.fetchWithRetry('/data');
  expect(result.status).toBe(200);
  expect(mockApi.callCount).toBe(3);
});

// ✅ GOOD: Test handler logic, not route existence
test('GET /users returns paginated results', async () => {
  const res = await request(app).get('/users?page=2&limit=10');
  expect(res.body.data).toHaveLength(10);
  expect(res.body.page).toBe(2);
});

// ✅ GOOD: Render with behavioral assertion
test('displays user name and email', () => {
  render(<UserProfile user={testUser} />);
  expect(screen.getByText(testUser.name)).toBeInTheDocument();
  expect(screen.getByText(testUser.email)).toBeInTheDocument();
});
```

### Gate Function

```
BEFORE writing any test:
  Ask: "What decision, transformation, or behavior does this test exercise?"

  IF answer is "checks a value is set/exists":
    STOP - This is a tautology test
    Find the logic that USES the value and test that instead

  IF answer is "confirms something is registered/configured":
    STOP - Test the behavior the registration enables

  Red flags:
    - Test mirrors a single line of source code
    - No conditional or transformation in the code under test
    - Test only verifies assignment or existence
    - Removing the test would never let a real bug through
```

## Anti-Pattern 7: Testing Test Code

**The violation:**
```typescript
// ❌ BAD: Writing tests for a test factory
describe('createMockUser', () => {
  test('returns user with default values', () => {
    const user = createMockUser();
    expect(user.id).toBeDefined();
    expect(user.name).toBeDefined();
    expect(user.email).toContain('@');
  });

  test('applies overrides', () => {
    const user = createMockUser({ name: 'Alice' });
    expect(user.name).toBe('Alice');
  });
});

// ❌ BAD: Testing a test fixture loader
test('loadFixture returns parsed JSON', () => {
  const data = loadFixture('users.json');
  expect(data).toBeInstanceOf(Array);
});
```

**Why this is wrong:**
- **Circular validation** -- test code verifying test code, no production behavior exercised
- **Inflates coverage on non-production code** -- meaningless metric improvement
- **Test code is validated implicitly** -- if `createMockUser` is broken, the production tests using it fail
- **If a helper is complex enough to need its own tests, it should be production code**

**The fix:**
```typescript
// ✅ GOOD: Test the production code that uses the factory
test('deactivated users cannot place orders', () => {
  const user = createMockUser({ status: 'deactivated' });
  const order = new OrderService();

  expect(() => order.place(user, cart)).toThrow('User is deactivated');
});

// The factory is validated by this test working correctly.
// If createMockUser breaks, this test breaks -- no separate test needed.
```

### Gate Function

```
BEFORE writing a test:
  Ask: "Is the code under test production code or test infrastructure?"

  IF test infrastructure (helpers, fixtures, factories, mocks, utilities):
    STOP - Don't write tests for test code
    Test code is validated when production tests that use it pass

  IF the helper is complex enough to seem like it needs tests:
    Move it to production code (shared library/utility)
    THEN write tests for it as production code
```

## When Mocks Become Too Complex

**Warning signs:**
- Mock setup longer than test logic
- Mocking everything to make test pass
- Mocks missing methods real components have
- Test breaks when mock changes

**your human partner's question:** "Do we need to be using a mock here?"

**Consider:** Integration tests with real components often simpler than complex mocks

## TDD Prevents These Anti-Patterns

**Why TDD helps:**
1. **Write test first** → Forces you to think about what you're actually testing
2. **Watch it fail** → Confirms test tests real behavior, not mocks
3. **Minimal implementation** → No test-only methods creep in
4. **Real dependencies** → You see what the test actually needs before mocking

**If you're testing mock behavior, you violated TDD** - you added mocks without watching test fail against real code first.

## Quick Reference

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

## Red Flags

- Assertion checks for `*-mock` test IDs
- Methods only called in test files
- Mock setup is >50% of test
- Test fails when you remove mock
- Can't explain why mock is needed
- Mocking "just to be safe"
- Tautology tests that restate source code as assertions
- No conditional or transformation in code under test
- Tests that only verify assignment or existence
- Tests targeting test helpers, factories, or fixtures instead of production code

## The Bottom Line

**Mocks are tools to isolate, not things to test.**

If TDD reveals you're testing mock behavior, you've gone wrong.

Fix: Test real behavior or question why you're mocking at all.
