# SDK Testing Strategies

Three-tier testing approach for API client SDKs: unit, contract, and integration.

## Unit Tests with HTTP Mocking

```python
# Python: respx (httpx mock)
import respx

@respx.mock
def test_get_user():
    respx.get("https://api.example.com/v1/users/123").mock(
        return_value=httpx.Response(200, json={"id": "123", "name": "Alice"})
    )
    client = ExampleClient(api_key="test")
    user = client.users.get("123")
    assert user.name == "Alice"

@respx.mock
def test_retry_on_503():
    route = respx.get("https://api.example.com/v1/users/123")
    route.side_effect = [
        httpx.Response(503),
        httpx.Response(200, json={"id": "123", "name": "Alice"}),
    ]
    client = ExampleClient(api_key="test")
    user = client.users.get("123")
    assert route.call_count == 2
```

## Contract Tests

Record API responses as JSON fixtures, then replay against the SDK to verify parsing.

```
tests/
  fixtures/
    get_user_200.json
    list_users_paginated.json
    create_user_422.json
```

- **Record**: capture real responses from staging/sandbox
- **Replay**: mock HTTP layer with fixtures
- **Assert**: SDK parses correctly, models match expected shapes
- **Maintain**: update fixtures when API schema changes

## Integration Tests

```python
# Run against staging environment
def test_integration_create_and_delete():
    client = ExampleClient(api_key=os.environ["STAGING_API_KEY"])
    user = client.users.create(name="Test", email="test@example.com")
    assert user.id
    client.users.delete(user.id)
```

**Integration test rules**:
- Gate behind env var (don't run in CI by default)
- Clean up created resources in teardown
- Use dedicated staging/sandbox environment
- Tolerate latency -- use generous timeouts

## What to Test

| Layer | Test Type | Coverage Target |
|-------|-----------|-----------------|
| Request building | Unit | URL construction, headers, query params |
| Response parsing | Unit + Contract | Model deserialization, edge cases |
| Retry logic | Unit | Backoff timing, max attempts, Retry-After |
| Pagination | Unit | Iterator behavior, cursor propagation |
| Error mapping | Unit | Status code to exception class |
| Auth flow | Unit | Token refresh, header injection |
| Full round-trip | Integration | Create/read/update/delete cycle |
