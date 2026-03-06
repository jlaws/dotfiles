# Swift Memory Management (ARC)

Retain cycle patterns, weak/unowned examples, and closure capture pitfalls.

## Common Retain Cycles

### Closures

```swift
// WRONG - self captured strongly
class ViewModel {
  var onUpdate: (() -> Void)?
  func setup() {
    onUpdate = {
      self.refresh()  // retain cycle
    }
  }
}

// RIGHT - weak self
class ViewModel {
  var onUpdate: (() -> Void)?
  func setup() {
    onUpdate = { [weak self] in
      self?.refresh()
    }
  }
}
```

### Delegates

```swift
// WRONG
protocol ViewDelegate: AnyObject {}
class View {
  var delegate: ViewDelegate?  // should be weak
}

// RIGHT
class View {
  weak var delegate: ViewDelegate?
}
```

### Two-way relationships

```swift
// WRONG
class Parent {
  var child: Child?
}
class Child {
  var parent: Parent?  // retain cycle
}

// RIGHT
class Parent {
  var child: Child?
}
class Child {
  weak var parent: Parent?
}
```

## Weak Self Dance

```swift
// WRONG - crashes if self deallocated
closure { [weak self] in
  guard let self else { return }
  self.doWork()
  await self.asyncWork()  // self might be nil now
}

// RIGHT - recapture or use local
closure { [weak self] in
  guard let self else { return }
  doWork()
  Task { [weak self] in
    await self?.asyncWork()
  }
}
```

## Escaping Closure Retain Cycles

```swift
// WRONG
class ViewModel {
  var completion: (() -> Void)?
  func start() {
    networkCall { [self] in  // captures self strongly
      self.completion?()
    }
  }
}

// RIGHT
class ViewModel {
  var completion: (() -> Void)?
  func start() {
    networkCall { [weak self] in
      self?.completion?()
    }
  }
}
```

## Performance Note

`unowned` < `weak` overhead (no optional check), but profile first. Premature optimization causes crashes.
