# Testing and Integration Architecture

Testing pyramid, integration patterns, test isolation, CI design, MCP testing, and gotchas.

## Testing Pyramid / Trophy

| Level | Ratio | Speed | Cost | What |
|-------|-------|-------|------|------|
| **Unit** | 50-60% | ~1ms | Low (mock everything) | Function logic, edge cases, single responsibility |
| **Integration** | 30-40% | ~100ms | Medium (real DB/services) | Components work together, DB transactions, API contracts |
| **End-to-End** | 5-10% | ~1-5s | High (full stack) | Happy path workflows, user scenarios, cross-system flows |

```python
# Unit: Test logic in isolation
def test_calculate_total():
    assert calculate_total([10, 20, 30]) == 60

# Integration: Test with real database
@pytest.mark.integration
def test_save_and_retrieve_user(db):
    user = User(name="Alice", email="alice@example.com")
    db.session.add(user)
    db.session.commit()

    fetched = db.session.query(User).filter_by(email="alice@example.com").first()
    assert fetched.name == "Alice"

# E2E: Test full flow
@pytest.mark.e2e
async def test_user_registration_flow(client):
    response = await client.post("/register", json={
        "name": "Bob", "email": "bob@example.com", "password": "secret"
    })
    assert response.status_code == 201

    login = await client.post("/login", json={
        "email": "bob@example.com", "password": "secret"
    })
    assert login.status_code == 200
```

**Rule**: Most tests should be unit; use integration for tricky boundaries; E2E sparingly.

## Integration Testing Patterns

### In-Memory Databases
```python
# SQLite in-memory for fast tests
@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    yield SessionLocal()

# PostgreSQL testcontainer (with Docker)
@pytest.fixture(scope="session")
def postgres():
    from testcontainers.postgres import PostgresContainer
    with PostgresContainer("postgres:15") as postgres:
        yield postgres.get_connection_url()
```

### API Contract Testing
```python
# Test that API matches expected schema (Pact, Testify)
import pact

consumer = pact.Consumer("MyApp")
provider = pact.Provider("MyAPI")

def test_get_user_contract():
    (consumer
        .upon_receiving("a request for user 123")
        .with_request("GET", "/users/123")
        .will_respond_with(200, body={
            "id": 123,
            "name": "Alice",
            "email": "alice@example.com"
        }))

    with consumer.pact_with(provider):
        response = requests.get("http://localhost:8080/users/123")
        assert response.json()["name"] == "Alice"
```

### Test Fixtures vs Factories
```python
# Fixture: Setup once per test session (expensive)
@pytest.fixture(scope="session")
def large_dataset():
    return generate_million_records()

# Factory: Create per test (fast, isolated)
@pytest.fixture
def user_factory():
    def _create(name="Alice", email="alice@example.com"):
        return User(name=name, email=email)
    return _create

def test_with_factory(user_factory):
    alice = user_factory()
    bob = user_factory(name="Bob")
```

## Test Isolation Strategies

```python
# 1. Database: Use transactions, rollback after each test
@pytest.fixture
def db_session():
    session = SessionLocal()
    session.begin_nested()  # Savepoint
    yield session
    session.rollback()

# 2. Mocking: Isolate external dependencies
from unittest.mock import patch, MagicMock

@patch("requests.get")
def test_fetch_user(mock_get):
    mock_get.return_value.json.return_value = {"id": 1, "name": "Alice"}
    user = fetch_user(1)
    assert user["name"] == "Alice"
    mock_get.assert_called_once_with("http://api.example.com/users/1")

# 3. Environment: Separate test env vars
@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
```

## CI Test Pipeline Design

```yaml
# Parallel stages for speed
pipeline:
  - unit-tests:
      parallel: 4
      timeout: 5m
      fail-fast: true  # Stop other jobs if this fails

  - integration-tests:
      dependencies: ["unit-tests"]  # Run after unit
      parallel: 2
      timeout: 15m
      requires-db: true

  - e2e-tests:
      dependencies: ["integration-tests"]
      parallel: 1  # Don't parallelize; single E2E flow
      timeout: 10m

  - coverage-report:
      dependencies: ["unit-tests", "integration-tests"]
      threshold: 80%  # Fail if coverage drops
```

**Gotcha**: Don't parallelize E2E tests; shared state causes flakes.

---

## MCP Testing and Integration

### Using the MCP Inspector

```bash
# Test stdio server interactively
npx @modelcontextprotocol/inspector python my_server.py

# Test remote server
npx @modelcontextprotocol/inspector http://localhost:8080
```

Always test with the MCP Inspector before integrating with a host. It shows the exact JSON-RPC messages exchanged, making protocol issues visible immediately.

### Programmatic Testing (Python)

```python
import pytest
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

@pytest.fixture
async def client():
    params = StdioServerParameters(command="python", args=["my_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

@pytest.mark.asyncio
async def test_search_tool(client):
    result = await client.call_tool("search_docs", {"query": "authentication"})
    assert result.content[0].type == "text"
    assert "auth" in result.content[0].text.lower()

@pytest.mark.asyncio
async def test_list_tools(client):
    tools = await client.list_tools()
    tool_names = [t.name for t in tools.tools]
    assert "search_docs" in tool_names
    assert "get_user" in tool_names

@pytest.mark.asyncio
async def test_resource(client):
    result = await client.read_resource("config://app")
    data = json.loads(result.contents[0].text)
    assert "database" in data
```

### Codex Integration

```json
# claude_desktop_config.json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["/absolute/path/to/my_server.py"],
      "env": {
        "DATABASE_URL": "postgresql://localhost/mydb"
      }
    },
    "remote-server": {
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

```json
# .mcp.json (Project-Level Config)
{
  "mcpServers": {
    "project-tools": {
      "command": "python",
      "args": ["./tools/mcp_server.py"],
      "env": {
        "PROJECT_ROOT": "."
      }
    }
  }
}
```

## Gotchas

- **Flaky tests**: Timeouts, race conditions, external API delays. Add `@pytest.mark.flaky(reruns=3)` for known flakes.
- **Test data creep**: Snapshot tests can become stale. Use `pytest --snapshot-update` carefully.
- **Mock abuse**: Mocking too much hides real bugs. Prefer integration tests for critical paths.
- **Shared test fixtures**: Scope too broad, tests interfere. Use `autouse=True` sparingly, prefer explicit injection.
- **E2E brittleness**: Browser automation, timing. Use explicit waits, not hardcoded sleeps.
