# Claude Code Configuration

## Author
- **Name**: Joe Laws
- **Email**: joe.laws@gmail.com
- **Primary**: Swift 6+, SwiftUI, iOS/macOS
- **Secondary**: TypeScript, Python, Go, Rust

---

## Communication Style

### Do
- Be concise and direct. No filler.
- Lead with the answer, explain after if needed.
- Use bullet points and code examples.
- Assume I'm an experienced developer.
- Challenge my assumptions when appropriate.
- Ask clarifying questions rather than guessing.

### Don't
- Over-explain basic concepts.
- Add unnecessary caveats or warnings.
- Repeat requirements back to me.
- Use excessive praise or encouragement.

---

## Code Style (Universal)

### Formatting
- **Indentation**: 2 spaces (no tabs)
- **Line endings**: LF (Unix)
- **Charset**: UTF-8 (no BOM)
- **Trailing whitespace**: Trim
- **Final newline**: Always
- **Line length**: 80-100 soft limit

### Naming
- Variables/functions: `camelCase` (JS/TS/Swift) or `snake_case` (Python/Rust/Go)
- Types/Classes: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE` or language idiom
- Booleans: `is`, `has`, `can`, `should` prefix
- Functions: verb prefix (`get`, `set`, `create`, `handle`)

### Organization
- Group imports: stdlib, third-party, local (blank lines between)
- One concept per file when practical
- Keep files under 300 lines
- Tests colocated or in parallel directory

---

## Development Philosophy

### TDD First
1. **Red**: Write failing test defining expected behavior
2. **Green**: Write minimal code to pass
3. **Refactor**: Clean up while tests stay green

### Clean Code
- **Single Responsibility**: One reason to change per function/class
- **DRY**: Extract duplicates, but don't over-abstract prematurely
- **YAGNI**: Don't build until needed
- **Composition over Inheritance**: Prefer protocols/interfaces
- **Explicit over Implicit**: Clarity beats cleverness

### Quality Standards
- No `any` in TypeScript (use `unknown`)
- No force unwraps in Swift (unless provably safe)
- All public APIs documented
- Error handling: explicit, typed, recoverable

### Refactoring Triggers
- Function > 30 lines → extract
- > 3 parameters → parameter object
- Nested conditionals > 2 levels → early returns
- Duplicated code > 2x → extract utility

---

## Git Workflow

### Branches
- **Main**: `main` (always deployable)
- **Features**: `feature/short-description`
- **Fixes**: `fix/issue-description`
- **Cleanup**: `cleanup/what-changed`
- **Docs**: `docs/what-documented`

### Commits
- Format: `type: description` (lowercase, imperative)
- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
- Keep atomic (one logical change)
- Describe what and why, not how

### PRs
- Title matches primary commit
- Include test plan
- Reference related issues
- Keep small and focused

---

## Swift & iOS

### Standards
- Swift 6+ with strict concurrency
- iOS 16+ minimum (latest minus 2)
- SwiftUI over UIKit for new views
- Swift Package Manager for dependencies

### Patterns
```swift
// Early exit with guard
guard let value = optional else { return }

// Async/await over completion handlers
func fetch() async throws -> Data

// Typed throws (Swift 6)
func process() throws(ValidationError)

// Value types preferred
struct User: Sendable, Codable, Identifiable { }

// @Observable over ObservableObject (iOS 17+)
@Observable final class ViewModel { }
```

### Testing
- XCTest with `@Test` macro (Swift Testing)
- One test file per source file
- Mock protocols, not concrete types

### Avoid
- Force unwraps (`!`) without validation
- Implicitly unwrapped optionals (except `@IBOutlet`)
- Massive view controllers
- `Any`/`AnyObject` when concrete types known
- Main thread blocking

---

## TypeScript & JavaScript

### Configuration
- `strict: true` always
- No implicit any
- Target ES2022+

### Patterns
```typescript
// Interfaces over type aliases for objects
interface User {
  id: string;
  name: string;
}

// const assertions for literals
const ROLES = ['admin', 'user'] as const;
type Role = typeof ROLES[number];

// unknown over any
function parse(input: unknown): Result { }

// Discriminated unions for state
type State =
  | { status: 'loading' }
  | { status: 'success'; data: Data }
  | { status: 'error'; error: Error };
```

### Frameworks
- React: Functional components with hooks only
- Next.js: App Router, React Server Components
- Testing: Vitest or Jest with Testing Library

### Avoid
- `any` type
- `enum` (use const objects or unions)
- Class components
- Default exports
- `var` keyword

---

## Python

### Standards
- Python 3.11+
- Type hints required for public functions
- Ruff for linting and formatting

### Patterns
```python
# Type hints required
def process(items: list[dict[str, Any]]) -> Result:
    ...

# Dataclasses or Pydantic
@dataclass
class User:
    id: str
    name: str

# pathlib over os.path
from pathlib import Path
config = Path.home() / ".config" / "app"

# Context managers for resources
async with aiohttp.ClientSession() as session:
    ...
```

### Testing
- pytest with fixtures
- Parametrize for multiple inputs
- pytest-asyncio for async tests

### Package Management
- Prefer `uv` for speed
- `poetry` for complex projects
- Pin dependencies in `pyproject.toml`

---

## Go

### Standards
- Go 1.21+
- Always use modules
- `gofmt`/`goimports` enforced

### Patterns
```go
// Explicit error handling
result, err := doSomething()
if err != nil {
    return fmt.Errorf("context: %w", err)
}

// Context for cancellation
func Process(ctx context.Context, data Data) error

// Table-driven tests
func TestProcess(t *testing.T) {
    tests := []struct{ name string; input Input; want Output }{...}
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {...})
    }
}
```

### Avoid
- Naked returns in functions > 5 lines
- Package-level variables (except errors)
- `panic` for recoverable errors
- Ignoring errors with `_`

---

## Rust

### Standards
- Edition 2021+
- All Clippy warnings addressed
- `rustfmt` enforced

### Patterns
```rust
// Result for fallible operations
fn process(data: &[u8]) -> Result<Output, ProcessError>

// &str over String for params
fn greet(name: &str) -> String

// ? for error propagation
let content = fs::read_to_string(path)?;

// Derive common traits
#[derive(Debug, Clone, PartialEq, Eq)]
struct Config { ... }
```

### Avoid
- `unwrap()`/`expect()` in library code
- `unsafe` without documented invariants
- `clone()` when borrowing suffices
- Stringly-typed APIs

---

## Security

### Secrets
- Never commit secrets, API keys, or credentials
- Use environment variables or secret managers
- Add sensitive patterns to `.gitignore`

### Input Handling
- Validate and sanitize all user input
- Parameterized queries only (no string concat for SQL)
- Escape output based on context

### Dependencies
- Audit before adding (`npm audit`, `cargo audit`)
- Keep updated for security patches
- Prefer well-maintained libraries

### Red Flags
- `eval()`, `exec()`, dynamic code execution
- Direct SQL string construction
- Disabled security features
- Hardcoded credentials

---

## Anti-Patterns

### Code
- Premature abstraction (wait for 2+ implementations)
- Over-engineering (start simple)
- God objects (split by responsibility)
- Deep nesting > 2-3 levels (early returns)
- Magic numbers/strings (use constants)

### Process
- Writing code before understanding the problem
- Skipping tests "to save time"
- Large PRs that are hard to review
- Catching exceptions without handling
- Copy-pasting without understanding

### Communication
- Vague commits ("fix stuff", "update code")
- Undocumented public APIs
- TODOs without context
- Commented-out code in codebase
