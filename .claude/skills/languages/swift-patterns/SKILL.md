---
name: swift-patterns
description: Use when creating Swift projects (iOS, macOS, CLI, frameworks), making memory management decisions, or working with Swift concurrency
---

# Swift Patterns

Tooling opinions, memory management, concurrency decisions, and non-linter-enforceable style rules.

## Style Guide

Source: Google Swift Style Guide + Swift API Design Guidelines. Only rules linters/formatters cannot enforce.

### Naming
- Treat acronyms as whole words: `loadHttpUrl` not `loadHTTPURL`
- Boolean names: `is`, `has`, `can`, `should` prefix
- Methods: imperative for side-effects (`sort()`), nouns for queries (`sorted()`)
- Mutating pairs: `sort()`/`sorted()`, `formUnion()`/`union()`
- Delegate patterns: `didFinish`, `willStart`, `shouldAllow` prefix
- File naming: `MyType+Protocol.swift` for extensions implementing protocols
- Error types: `Error` suffix (`NetworkError`, `ValidationError`)
- Avoid getters/setters: computed properties, not `getX()` methods

### Practices
- Force unwrap `!` requires safety comment explaining why nil is impossible
- `guard` over nested `if` for early exits
- Protocol naming: nouns (`Collection`) vs `-able`/`-ible`/`-ing` (`Equatable`, `Copying`)
- No `Any` without runtime check immediately after
- Error handling: throw custom error types, not `Optional` for recoverable failures
- Comments explain WHY, not WHAT
- Max function body ~40 lines

### SwiftLint Essential Rules (non-formatter)
```yaml
opt_in_rules:
  - explicit_init
  - explicit_type_interface
  - force_unwrapping
  - implicitly_unwrapped_optional
line_length: 120
function_body_length: 40
```

## Tooling Defaults

| Concern | Use | Why |
|---------|-----|-----|
| Package manager | SPM (Swift Package Manager) | Native, integrated, zero config |
| Linter | SwiftLint | De facto standard |
| Formatter | swift-format | Official Apple tool |
| Testing | XCTest or Swift Testing (5.9+) | Built-in, mature ecosystem |
| Dependencies | SPM only | Avoid CocoaPods/Carthage complexity |

### swift-format config
```json
{
  "version": 1,
  "lineLength": 120,
  "indentation": { "spaces": 2 },
  "respectsExistingLineBreaks": true
}
```

## Project Types

| Type | When | Structure |
|------|------|-----------|
| **iOS App** | iPhone/iPad apps | Xcode project + SPM for deps |
| **macOS App** | Desktop apps | Xcode project + AppKit/SwiftUI |
| **Framework** | Reusable libraries | SPM-only, no Xcode project |
| **CLI** | Command-line tools | SPM with executable target |
| **Multi-platform** | iOS + macOS + watchOS | Shared code in SPM package |

### Framework Package.swift
```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "MyLibrary",
  platforms: [.iOS(.v16), .macOS(.v13)],
  products: [
    .library(name: "MyLibrary", targets: ["MyLibrary"])
  ],
  dependencies: [],
  targets: [
    .target(name: "MyLibrary"),
    .testTarget(name: "MyLibraryTests", dependencies: ["MyLibrary"])
  ]
)
```

### Directory Structure
```
Sources/MyLibrary/
  MyLibrary.swift
  Models/
  Views/
  Utilities/
Tests/MyLibraryTests/
  MyLibraryTests.swift
Package.swift
.swiftlint.yml
```

## Memory Management (ARC)

### Weak vs Unowned Decision

| Use `weak` | Use `unowned` | Use strong |
|------------|---------------|------------|
| Optional relationship (may outlive) | Non-optional, guaranteed lifetime | Ownership responsibility |
| Parent-child with independent lifecycle | Child-parent where child can't outlive | Default case |
| Delegate patterns | Closure capturing guaranteed-alive context | Value types (struct, enum) |

**Rule**: Start with `weak`, switch to `unowned` only after profiling shows measurable overhead.

For detailed retain cycle examples and the weak self dance, see [references/memory-management.md](references/memory-management.md).

## Concurrency Patterns

### async/await vs GCD Decision

| Use async/await | Use GCD |
|-----------------|---------|
| Default for all new code | Legacy code integration |
| Structured concurrency needed | Fire-and-forget tasks |
| Error propagation matters | Very low-level control required |

**Rule**: Always prefer async/await unless integrating with legacy GCD code.

For @MainActor, actors, TaskGroup, AsyncStream, and cancellation examples, see [references/concurrency-patterns.md](references/concurrency-patterns.md).

## Common Gotchas

| Gotcha | Rule |
|--------|------|
| Force unwraps | `guard let` or `if let`, never bare `!` without safety comment |
| Implicitly unwrapped optionals | Use `Optional`, reserve `!` for IBOutlets only |
| Blocking main actor | Offload heavy work via `Task.detached` |
| Actor reentrancy | Keep actor methods synchronous when mutating state |
| Sendable violations | Use structs or `@unchecked Sendable` for class types |
| Task cancellation | Always `try Task.checkCancellation()` in loops |
| Weak self in async closures | Re-capture `[weak self]` after each suspension point |
| Array/Dict copy-on-write | Value types but COW-optimized; copy happens on mutation |
| Protocol with associated types | Cannot use as existential; use generics or type erasure |

### Struct vs Class Decision

| Use struct | Use class |
|------------|-----------|
| Value semantics needed | Reference semantics needed |
| Immutable data models | Lifecycle management (deinit) |
| No inheritance | Inheritance required |
| Small, simple types | Complex objects with identity |

## Testing & Error Handling

For XCTest, Swift Testing, dependency injection, Result type, and custom error patterns, see [references/testing-and-errors.md](references/testing-and-errors.md).

## Performance

For lazy properties, custom COW, @inline hints, and Instruments profiling guidance, see [references/performance-tips.md](references/performance-tips.md).
