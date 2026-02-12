---
name: language-testing-patterns
description: Use when designing test suites, choosing fixtures/mocking strategies, or implementing language-specific test patterns. Opinionated frameworks and anti-patterns for Python and JavaScript/TypeScript.
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

## Python Testing (pytest)

### Framework Selection
- **Default**: pytest with `pytest-asyncio` (mode = "auto"), `pytest-cov`
- Use Hypothesis for property-based testing (invariants, parsers, roundtrips)

### Fixture Scope Selection
| Scope | Use For | Gotcha |
|-------|---------|--------|
| `function` (default) | Mutable state, DB sessions | Safe but slow if expensive |
| `module` | Expensive read-only resources | Shared state leaks between tests |
| `session` | Config, DB engine creation | Never for mutable data |

**Rule**: Use narrowest scope that doesn't kill performance. When in doubt, use `function`.

### Fixture Composition Over Inheritance
```python
@pytest.fixture
def db_session(db_engine):
    session = Session(db_engine)
    yield session
    session.rollback()  # Fast cleanup, not commit()
    session.close()

@pytest.fixture
def user(db_session):
    user = UserFactory.create()
    db_session.add(user)
    db_session.flush()
    return user
```

- Chain fixtures via dependency injection, not class inheritance
- Always `yield` + cleanup, not just `return`
- DB sessions should `rollback()` not `commit()` -- faster, auto-cleans

### `autouse` Sparingly
- Only for truly universal setup (e.g., resetting a global clock)
- Invisible dependencies make tests harder to understand
- Prefer explicit fixture parameters

### Mocking Opinion: `monkeypatch` > `unittest.mock`
```python
# Prefer monkeypatch (auto-reverts)
def test_api_call(monkeypatch):
    monkeypatch.setattr("myapp.services.requests.get", lambda: MockResponse())
    monkeypatch.setenv("API_KEY", "test-key")
```

**Patch where it's used, not where it's defined**. This is the #1 mock mistake.
```python
# Module: myapp/services.py imports requests
# WRONG: @patch("requests.get")
# RIGHT: @patch("myapp.services.requests.get")
```

### When to Use `MagicMock` vs `Mock`
- `MagicMock`: when code uses dunder methods (`__len__`, `__iter__`)
- `Mock`: default choice, simpler, fewer implicit behaviors
- `spec=True`: always set when mocking classes -- catches typos in attribute access

### Parametrize Decisions
- **Use when**: Same logic, different inputs (validation rules, edge cases, matrix testing)
- **Avoid when**: Different test logic per case (just write separate tests), >10 sets (use Hypothesis)
- Always use `id=` for readable test output

```python
@pytest.mark.parametrize("input,expected", [
    pytest.param("valid@email.com", True, id="valid-email"),
    pytest.param("no-at-sign", False, id="missing-at"),
])
def test_email_validation(input, expected):
    assert is_valid_email(input) == expected
```

### Test Organization
```
tests/
  conftest.py              # Shared fixtures (session/module scope)
  unit/
    conftest.py            # Unit-test-specific fixtures
    test_services.py
  integration/
    conftest.py            # DB setup, API clients
    test_api.py
  factories.py             # Factory Boy or manual factories
```

**conftest.py Strategy**:
- Root: DB engine, app config, shared factories
- Directory-level: scope-specific fixtures
- Never import from conftest -- pytest injects automatically

### CI Markers
```ini
# pyproject.toml
[tool.pytest.ini_options]
markers = ["slow: marks slow tests", "integration: marks integration tests"]
addopts = "--strict-markers --tb=short -q --cov-fail-under=80"
```

- Use `--strict-markers` to catch typos
- Run `pytest -m "not integration"` in pre-commit, full suite in CI

## JavaScript/TypeScript Testing (Vitest/Jest)

### Framework Selection
- **Vitest**: Default for Vite projects, ESM-native, fast HMR
- **Jest**: Mature ecosystem for non-Vite projects, use `ts-jest` or SWC transform
- Near-identical APIs, migration is low-effort

### Dependency Injection Over Module Mocking
```typescript
// Prefer: inject dependencies for testability
class UserService {
  constructor(private repo: IUserRepository) {}
}

// Avoid: vi.mock('module') -- brittle, breaks on refactors
```

**Module mocking (`vi.mock`, `jest.mock`) is a last resort**. It couples tests to import paths and breaks when files move.

### Testing Async Properly
- Always `await` assertions on promises: `await expect(fn()).rejects.toThrow()`
- Never use `done()` callbacks -- use async/await
- Mock timers with `vi.useFakeTimers()`, clean up with `vi.useRealTimers()`

### Frontend Component Testing
- **Query priority**: `getByRole` > `getByLabelText` > `getByPlaceholderText` > `getByTestId`
- `data-testid` is a last resort, not first choice
- Use `userEvent` over `fireEvent` -- simulates real behavior (focus, blur, etc.)
- Test what user sees, not component internals
- Avoid snapshot tests for components -- catch everything and nothing

### Integration Test Boundaries
- API integration: use `supertest` with real app + test database
- `beforeEach`: truncate tables, not drop/create (faster)
- Test full request/response cycle including middleware
- Separate integration tests with markers/directories, run separately in CI

### Mocking Strategies
| Scenario | Approach |
|----------|----------|
| External APIs | `msw` (Mock Service Worker) -- intercepts at network level |
| Database | Test containers or in-memory DB |
| Time/dates | `vi.useFakeTimers()` |
| Modules | DI first; `vi.mock()` only if no other option |
| Environment vars | `vi.stubEnv()` or monkeypatch |

### Mock Hygiene
- `vi.clearAllMocks()` in `beforeEach`, not `afterEach`
- Prefer `mockResolvedValueOnce` over `mockResolvedValue` -- forces explicit setup per test
- Verify with `toHaveBeenCalledWith`, not just `toHaveBeenCalled`

### Test Organization
```
src/
  services/
    user.service.ts
    user.service.test.ts     # Co-located unit tests
tests/
  integration/               # Separate integration tests
  fixtures/                  # Shared factories
  setup.ts                   # Global test setup
```

- Co-locate unit tests with source files
- Separate integration/e2e tests into dedicated directories
- Share fixtures via `fixtures/`, not copy-paste

## Test Generation Patterns

### Naming Convention
`test_{function}_{scenario}_{expected_result}`

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
