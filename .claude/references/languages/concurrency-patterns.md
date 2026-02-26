# Swift Concurrency Patterns

Deep-dive into async/await, actors, tasks, and structured concurrency.

## @MainActor Usage

```swift
// Entire class
@MainActor
class ViewModel {
  var items: [Item] = []
  func update() { /* runs on main thread */ }
}

// Individual methods
class ViewModel {
  @MainActor
  func updateUI() { /* main thread */ }

  func fetchData() async { /* background */ }
}

// SwiftUI views (implicit @MainActor on body)
struct ContentView: View {
  var body: some View { /* always main thread */ }
}
```

## Actor Patterns

```swift
// Isolate mutable state
actor DataCache {
  private var cache: [String: Data] = [:]

  func get(_ key: String) -> Data? { cache[key] }
  func set(_ key: String, _ value: Data) { cache[key] = value }
}

// Usage
let cache = DataCache()
await cache.set("key", data)
let value = await cache.get("key")
```

## Task and TaskGroup

```swift
// Single task
Task {
  let data = await fetchData()
  await MainActor.run { updateUI(data) }
}

// Parallel tasks
await withThrowingTaskGroup(of: Data.self) { group in
  for url in urls {
    group.addTask { try await fetch(url) }
  }
  var results: [Data] = []
  for try await result in group {
    results.append(result)
  }
  return results
}
```

## AsyncStream for Event Streams

```swift
func locationUpdates() -> AsyncStream<Location> {
  AsyncStream { continuation in
    let manager = LocationManager()
    manager.onUpdate = { location in
      continuation.yield(location)
    }
    continuation.onTermination = { _ in
      manager.stop()
    }
    manager.start()
  }
}

// Usage
for await location in locationUpdates() {
  print(location)
}
```

## Task Cancellation

```swift
let task = Task {
  for i in 1...100 {
    try Task.checkCancellation()  // throws CancellationError
    await doWork(i)
  }
}

// Later
task.cancel()
```

## Concurrency Gotchas

### Blocking main actor

```swift
// WRONG - freezes UI
@MainActor
func load() {
  let data = heavyComputation()  // blocks main thread
}

// RIGHT - offload work
@MainActor
func load() async {
  let data = await Task.detached {
    heavyComputation()
  }.value
}
```

### Actor reentrancy

```swift
actor Counter {
  var value = 0

  func increment() async {
    let old = value  // suspension point
    await Task.sleep(1_000_000_000)
    value = old + 1  // WRONG - value may have changed
  }
}

// RIGHT - atomic operation
actor Counter {
  var value = 0
  func increment() { value += 1 }
}
```

### Sendable violations

```swift
// WRONG - class not Sendable
class Data {}
Task {
  let data = Data()  // compiler error in strict concurrency
}

// RIGHT - use struct or mark @unchecked
struct Data: Sendable {}
```

### Task cancellation not automatic

```swift
// WRONG - continues even if task cancelled
Task {
  for i in 1...100 {
    await fetch(i)
  }
}

// RIGHT - check cancellation
Task {
  for i in 1...100 {
    try Task.checkCancellation()
    await fetch(i)
  }
}
```
