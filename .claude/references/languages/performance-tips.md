# Swift Performance Tips

Lazy properties, copy-on-write, inlining, and profiling.

## Lazy Properties

```swift
class ViewModel {
  lazy var formatter: DateFormatter = {
    let f = DateFormatter()
    f.dateStyle = .long
    return f
  }()
}
```

## Copy-on-Write

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
```

## @inline Hints

```swift
@inline(__always) func critical() { }
@inline(never) func debug() { }
```

## Profile with Instruments

- **Time Profiler**: CPU hotspots
- **Allocations**: memory usage
- **Leaks**: retain cycles
- **System Trace**: concurrency issues
