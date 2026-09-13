# Swift Testing and Error Handling Patterns

Test frameworks, dependency injection, and error type design.

## XCTest

```swift
import XCTest
@testable import MyLibrary

final class MyTests: XCTestCase {
  func testExample() {
    let result = compute(5)
    XCTAssertEqual(result, 10)
  }

  func testAsync() async throws {
    let data = try await fetchData()
    XCTAssertNotNil(data)
  }
}
```

## Swift Testing (5.9+)

```swift
import Testing
@testable import MyLibrary

@Test func computation() {
  #expect(compute(5) == 10)
}

@Test func asyncOperation() async throws {
  let data = try await fetchData()
  #expect(data != nil)
}
```

## Dependency Injection for Testability

```swift
// Production
protocol NetworkService {
  func fetch() async throws -> Data
}

class ViewModel {
  private let network: NetworkService
  init(network: NetworkService) {
    self.network = network
  }
}

// Test
class MockNetwork: NetworkService {
  func fetch() async throws -> Data { Data() }
}

let vm = ViewModel(network: MockNetwork())
```

## Custom Error Types

```swift
enum NetworkError: Error {
  case invalidURL
  case timeout
  case serverError(statusCode: Int)
}

func fetch() throws -> Data {
  throw NetworkError.timeout
}
```

## LocalizedError for User-Facing Messages

```swift
enum ValidationError: LocalizedError {
  case tooShort
  case invalidFormat

  var errorDescription: String? {
    switch self {
    case .tooShort: "Input is too short"
    case .invalidFormat: "Invalid format"
    }
  }
}
```
