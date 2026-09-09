---
name: language-testing-patterns
description: "Use when designing Python or JavaScript test suites."
---

# Language Testing Patterns

## Universal Principles

### What to Unit Test
- Pure functions, transformations, business logic
- Complex conditionals and state transitions
- Error handling paths
- Edge cases: empty arrays, null/undefined, boundary values

### What NOT to Unit Test
- Simple getters/setters, pass-through functions
- Framework internals (React rendering, Express routing)
- Implementation details -- test behavior, not structure
- Config/settings values (defaults, env var assignments, constants)
- Constructor assignments (`this.x = x` tests the language, not your code)
- Route/endpoint registration (test handler logic instead)
- Enum values and constants
- "Renders without crashing" with no behavior assertion
- Test code (test helpers, fixtures, factories, mocks, test utilities)
- Wiring/glue code with no logic

**Every test must exercise a decision point, transformation, or behavior path.**

### Test User Stories, Not Internals
- Focus tests on verifying key **user stories / user needs**, not implementation details
- Test **public interfaces / APIs** -- not private methods or internal state
- Coverage hierarchy: **important user story coverage > branch coverage > line coverage**
- Write a failing test for user-reported bugs **before** fixing
- Avoid testing trivial functionality (framework-generated getters/setters, `@ConfigurationProperties` classes, constructor assignments)

### Coverage Opinion
- 80% line coverage as gate, focus on branch coverage for business logic
- High coverage != well-tested. Missing edge cases matters more than line count.
- Exclude: `.d.ts`, config files, generated code, migrations, `__repr__`, `if TYPE_CHECKING`, test files, test helpers, test factories

### Factory Fixtures Over Inline Data
```python
# Python with faker
@pytest.fixture
def make_user(db_session):
    def _make_user(**kwargs):
        user = UserFactory.build(**kwargs)
        db_session.add(user)
        db_session.flush()
        return user
    return _make_user
```

```typescript
// JavaScript with faker
function createUser(overrides?: Partial<User>): User {
  return {
    id: faker.string.uuid(),
    name: faker.person.fullName(),
    email: faker.internet.email(),
    ...overrides,
  };
}
```

**Why**: Returns a callable -- tests create exactly what they need. Avoids "magic values" scattered across tests.

## Testing Pyramid (Shift Left)

```
        /  E2E  \          Expensive, slow, run infrequently (release testing)
       /----------\
      / Integration \       Moderate cost, run in CI
     /----------------\
    /    Unit Tests     \   Cheap, fast, run early and often
   /____________________\
```

- **Unit tests**: Bulk of test coverage. Fast, isolated, catch logic errors early
- **Integration tests**: Verify component interactions (DB, APIs, message queues). Run in CI
- **E2E tests**: Validate key user stories end-to-end. Most expensive, run for release verification
- **Shift left**: Identify defects as early as possible where they're cheapest to fix
- Rule of thumb: if a bug can be caught by a unit test, don't rely on integration/E2E to find it

## Ship Test Utilities with Components

When writing libraries or shared components, provide test utilities that make it easy for consumers to test:

- **In-memory fakes / test doubles** for your classes (e.g., `InMemoryUserRepository` alongside `UserRepository`)
- **Context managers / test fixtures** (Python: pytest fixtures; JS: setup helpers) to auto-configure test doubles
- **Spring Boot**: provide auto-configuration for test doubles via `@TestConfiguration`
- **Why**: lowers the barrier for consumers to write tests, promotes uniformity in testing patterns across the codebase

## Language-Specific Patterns

For detailed language-specific patterns, see the corresponding reference files:

- **Python (pytest)**: See `references/testing/python-testing-patterns.md` -- fixtures, monkeypatch, parametrize, conftest strategy, CI markers
- **JavaScript/TypeScript (Vitest/Jest)**: See `references/testing/javascript-testing-patterns.md` -- DI over module mocking, async testing, component testing, msw, mock hygiene

### Python Quick Reference
- pytest + pytest-asyncio + pytest-cov
- `monkeypatch` > `unittest.mock` (auto-reverts)
- Patch where it's used, not where it's defined
- Always use `spec=True` when mocking classes
- `yield` + cleanup in fixtures, `rollback()` not `commit()`

### JS/TS Quick Reference
- Vitest for Vite projects, Jest otherwise
- DI > module mocking (`vi.mock` is a last resort)
- `userEvent` > `fireEvent`, `getByRole` > `getByTestId`
- Always `await` async assertions
- `vi.clearAllMocks()` in `beforeEach`, not `afterEach`

## Test Generation Patterns

### Naming Convention

Test names describe **behavior**, not implementation:

| Pattern | Example |
|---|---|
| `should [behavior] when [condition]` | `should_reject_login_when_password_expired` |
| `test_{function}_{scenario}_{expected}` | `test_calculate_discount_bulk_order_20pct` |

Avoid: `test_method_name`, `testCase1`, names referencing internal method names.

### Arrange-Act-Assert Structure
```python
def test_user_creation_with_valid_data():
    # Arrange
    data = {"name": "Alice", "email": "alice@example.com"}

    # Act
    user = create_user(data)

    # Assert
    assert user.name == "Alice"
    assert user.email == "alice@example.com"
```

### Coverage Gap Detection Workflow
1. Run coverage: `pytest --cov=src --cov-report=json`
2. Parse JSON for `missing_lines` per file
3. Prioritize by complexity: branches > lines, business logic > utils
4. Generate tests for uncovered paths

### Mock Generation
```python
@pytest.fixture
def mock_api_client():
    mock = Mock(spec=APIClient)
    mock.fetch.return_value = {"status": "ok"}
    return mock
```

- Always use `spec=` to catch attribute errors
- Return realistic data shapes, not `"mocked_result"`

## Gotchas

### Python
- Fixture scope leaks: module/session fixtures with mutable state
- `autouse` fixtures create invisible dependencies
- Patching at wrong location (where defined vs. where used)
- Missing `yield` in fixtures (cleanup never runs)
- High coverage on `tests/` directory (meaningless, exclude it)

### JavaScript
- Using `fireEvent` instead of `userEvent` (misses real interactions)
- Snapshot tests for components (maintenance burden, no value)
- Module mocking when DI would work (breaks on refactors)
- Not awaiting async assertions (tests pass when they shouldn't)
- `data-testid` as first choice (tests implementation, not behavior)
