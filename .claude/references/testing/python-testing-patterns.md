# Python Testing Patterns (pytest)

Reference for Python-specific test patterns, fixtures, and mocking strategies. Extracted from `language-testing-patterns` skill.

## Framework Selection
- **Default**: pytest with `pytest-asyncio` (mode = "auto"), `pytest-cov`
- Use Hypothesis for property-based testing (invariants, parsers, roundtrips)

## Fixture Scope Selection
| Scope | Use For | Gotcha |
|-------|---------|--------|
| `function` (default) | Mutable state, DB sessions | Safe but slow if expensive |
| `module` | Expensive read-only resources | Shared state leaks between tests |
| `session` | Config, DB engine creation | Never for mutable data |

**Rule**: Use narrowest scope that doesn't kill performance. When in doubt, use `function`.

## Fixture Composition Over Inheritance
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

## `autouse` Sparingly
- Only for truly universal setup (e.g., resetting a global clock)
- Invisible dependencies make tests harder to understand
- Prefer explicit fixture parameters

## Mocking Opinion: `monkeypatch` > `unittest.mock`
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

## When to Use `MagicMock` vs `Mock`
- `MagicMock`: when code uses dunder methods (`__len__`, `__iter__`)
- `Mock`: default choice, simpler, fewer implicit behaviors

## Parametrize Decisions
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

## Test Organization
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

## CI Markers
```ini
# pyproject.toml
[tool.pytest.ini_options]
markers = ["slow: marks slow tests", "integration: marks integration tests"]
addopts = "--strict-markers --tb=short -q --cov-fail-under=80"
```

- Use `--strict-markers` to catch typos
- Run `pytest -m "not integration"` in pre-commit, full suite in CI
