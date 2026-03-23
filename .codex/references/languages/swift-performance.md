# Swift Performance Tips

Lazy properties, copy-on-write, inlining, compiler optimization, value vs reference types, string/collection performance, concurrency, and SwiftUI rendering.

## Compiler Optimization Flags

| Flag | Use Case | Tradeoff |
|------|----------|----------|
| `-Onone` | Debug builds, development | No optimization, fast compile, easier debugging |
| `-O` | Release builds, general | 30-60% faster, slower compile, harder debug symbols |
| `-Osize` | Binary size critical (embedded, watch) | Smaller code, slower than `-O`, still faster than debug |
| `-Ounchecked` | Math-heavy, bounds checks overhead | Array/division bounds unchecked, crashes on overflow |

```bash
# In Build Settings: Optimization Level = Optimize for Speed [-O]
# For math-heavy: ONE_FILE_COMPILE=NO swiftc -Ounchecked
```

**Gotcha**: `-Ounchecked` disables array bounds checks; use only if you've proven bounds safety.

## Lazy Properties

```swift
class ViewModel {
  lazy var formatter: DateFormatter = {
    let f = DateFormatter()
    f.dateStyle = .long
    return f
  }()
}
// Initialization deferred until first access; useful for expensive setup
```

## Copy-on-Write (Value Types)

```swift
// Arrays, dictionaries, sets use COW automatically
// Custom types:
struct MyCollection {
  private var storage: NSMutableArray

  mutating func append(_ item: Any) {
    if !isKnownUniquelyReferenced(&storage) {
      storage = storage.mutableCopy() as! NSMutableArray
    }
    storage.add(item)
  }
}
// Pass by value, copy only when modified
```

## Value Types vs Reference Types

| Dimension | Value Type (struct) | Reference Type (class) |
|-----------|---------------------|----------------------|
| Copy cost | O(size), often optimized to move | O(1) reference count |
| Mutability | Immutable by default, explicit `mutating` | Mutable by default |
| Memory | Stack (fast), or heap via COW | Heap allocation + refcounting |
| Thread safety | Isolated by default | Requires synchronization |
| Best for | Coordinates, small models, collections | View models, singletons, networking |

**Rule**: Use struct by default (immutable, stack-friendly); use class for identity, mutability, or ref semantics.

## String Performance

```swift
// SLOW: O(n) character-by-character
var result = ""
for char in largeString {
  result += String(char)  // repeated allocations
}

// FAST: Use contiguous storage
let result = largeString.map { String($0) }.joined(separator: "")

// UTF-8 Bridging (CFString overhead):
let swift = "hello"
let cfStr: CFString = swift as CFString  // toll-free bridge, O(1)
```

- Avoid `+=` loops; use `map().joined()` or `StringBuilder`-like approach.
- `String.Index` navigation is O(n) in UTF-8; cache indices if looping.

## Collection Performance

```swift
// FAST: Known size, contiguous memory
var nums = ContiguousArray<Int>(capacity: 1000)
nums.reserveCapacity(1000)  // avoid resizing
for i in 0..<1000 {
  nums.append(i)
}

// Array vs ContiguousArray: ContiguousArray skips bridging checks
let arr: [Int] = [1, 2, 3]  // may have Objective-C bridge overhead
let contArr = ContiguousArray(arr)  // pure Swift, no bridge

// Set lookup O(1) vs Array O(n)
let set = Set(arr)
if set.contains(2) { }  // instant
```

- Use `ContiguousArray` if you know you won't need Objective-C interop.
- Always call `reserveCapacity()` if size is known upfront.

## @inline Hints

```swift
@inline(__always) func critical() {
  // Force inlining for hot functions (measure first)
}
@inline(never) func debug() {
  // Prevent inlining; useful for profiler clarity
}
// Note: Compiler may ignore hints; use only after profiling proves benefit
```

## SwiftUI Performance

### View Identity & Body Invalidation
```swift
// BAD: body re-computes entire view tree on ANY @State change
struct ContentView: View {
  @State var counter = 0
  var body: some View {
    VStack {
      Text("\(counter)")
      ExpensiveView()  // re-rendered even though counter didn't affect it
    }
  }
}

// GOOD: Extract expensive subview, pass only needed state
struct ContentView: View {
  @State var counter = 0
  var body: some View {
    VStack {
      CounterText(counter: $counter)
      ExpensiveView()  // only re-rendered if its props change
    }
  }
}

struct CounterText: View {
  @Binding var counter: Int
  var body: some View { Text("\(counter)") }
}
```

### @State vs @StateObject
```swift
// @State: Value type, lost on parent re-render
@State var vm = ViewModel()  // re-created each render

// @StateObject: Preserved across re-renders
@StateObject var vm = ViewModel()  // created once, survives parent updates
```

Use `@StateObject` for view models that hold identity or expensive initialization.

## Concurrency Performance

```swift
// Actor isolation cost: queue hops on every property access
actor UserCache {
  var users: [User] = []

  func getUser(_ id: Int) -> User? {
    users.first { $0.id == id }  // crosses actor boundary
  }
}

// Optimize: batch reads, reduce boundary crossings
actor UserCache {
  var users: [User] = []

  func getUsers(_ ids: [Int]) -> [User] {
    ids.compactMap { id in users.first { $0.id == id } }  // one boundary crossing
  }
}

// Sendable overhead: type-checking, enforces value semantics
let data: some Sendable = (1, "hello")  // tuple is Sendable, zero runtime cost
```

- Reduce actor boundary crossings; batch operations when possible.
- `Sendable` is compile-time enforced; zero runtime cost.

## Profile with Instruments

- **Time Profiler**: CPU hotspots, call stacks
- **Allocations**: memory usage, growth over time
- **Leaks**: retain cycles, dangling references
- **System Trace**: concurrency issues, actor contention
- **Core Animation**: SwiftUI render performance, dropped frames
