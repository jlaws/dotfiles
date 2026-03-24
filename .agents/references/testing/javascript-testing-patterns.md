# JavaScript/TypeScript Testing Patterns (Vitest/Jest)

Reference for JS/TS-specific test patterns, fixtures, and mocking strategies. Extracted from `language-testing-patterns` skill.

## Framework Selection
- **Vitest**: Default for Vite projects, ESM-native, fast HMR
- **Jest**: Mature ecosystem for non-Vite projects, use `ts-jest` or SWC transform
- Near-identical APIs, migration is low-effort

## Dependency Injection Over Module Mocking
```typescript
// Prefer: inject dependencies for testability
class UserService {
  constructor(private repo: IUserRepository) {}
}

// Avoid: vi.mock('module') -- brittle, breaks on refactors
```

**Module mocking (`vi.mock`, `jest.mock`) is a last resort**. It couples tests to import paths and breaks when files move.

## Testing Async Properly
- Always `await` assertions on promises: `await expect(fn()).rejects.toThrow()`
- Never use `done()` callbacks -- use async/await
- Mock timers with `vi.useFakeTimers()`, clean up with `vi.useRealTimers()`

## Frontend Component Testing
- **Query priority**: `getByRole` > `getByLabelText` > `getByPlaceholderText` > `getByTestId`
- `data-testid` is a last resort, not first choice
- Use `userEvent` over `fireEvent` -- simulates real behavior (focus, blur, etc.)
- Test what user sees, not component internals
- Avoid snapshot tests for components -- catch everything and nothing

## Integration Test Boundaries
- API integration: use `supertest` with real app + test database
- `beforeEach`: truncate tables, not drop/create (faster)
- Test full request/response cycle including middleware
- Separate integration tests with markers/directories, run separately in CI

## Mocking Strategies
| Scenario | Approach |
|----------|----------|
| External APIs | `msw` (Mock Service Worker) -- intercepts at network level |
| Database | Test containers or in-memory DB |
| Time/dates | `vi.useFakeTimers()` |
| Modules | DI first; `vi.mock()` only if no other option |
| Environment vars | `vi.stubEnv()` or monkeypatch |

## Mock Hygiene
- `vi.clearAllMocks()` in `beforeEach`, not `afterEach`
- Prefer `mockResolvedValueOnce` over `mockResolvedValue` -- forces explicit setup per test
- Verify with `toHaveBeenCalledWith`, not just `toHaveBeenCalled`

## Test Organization
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

## Gotchas
- Using `fireEvent` instead of `userEvent` (misses real interactions)
- Snapshot tests for components (maintenance burden, no value)
- Module mocking when DI would work (breaks on refactors)
- Not awaiting async assertions (tests pass when they shouldn't)
- `data-testid` as first choice (tests implementation, not behavior)
