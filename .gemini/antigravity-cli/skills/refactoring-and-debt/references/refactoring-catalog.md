# Refactoring Catalog — Code Examples

## Extract Function / Method

**When**: Code block needs a comment to explain intent, or is duplicated.

```python
# Before
def process_order(order):
    # Calculate discount
    if order.customer.is_premium and order.total > 100:
        discount = order.total * 0.15
    elif order.total > 200:
        discount = order.total * 0.10
    else:
        discount = 0
    order.total -= discount
    # ... more processing

# After
def calculate_discount(order):
    if order.customer.is_premium and order.total > 100:
        return order.total * 0.15
    if order.total > 200:
        return order.total * 0.10
    return 0

def process_order(order):
    order.total -= calculate_discount(order)
    # ... more processing
```

## Extract Class / Module

**When**: A class has multiple responsibilities or a module is >300 lines with distinct sections.

Split by responsibility. Each resulting unit should have a single reason to change.

## Inline Function / Variable

**When**: Indirection adds no clarity. The function body is as clear as the name.

```typescript
// Before
function isEligible(age: number): boolean {
    return age >= 18;
}
const eligible = isEligible(user.age);

// After (if only used once and meaning is obvious)
const eligible = user.age >= 18;
```

## Replace Conditional with Polymorphism

**When**: Switch/if-else chain on a type field that appears in 3+ places.

```typescript
// Before
function getArea(shape: Shape): number {
    switch (shape.type) {
        case 'circle': return Math.PI * shape.radius ** 2;
        case 'rectangle': return shape.width * shape.height;
        case 'triangle': return 0.5 * shape.base * shape.height;
    }
}

// After
interface Shape {
    getArea(): number;
}
class Circle implements Shape {
    getArea() { return Math.PI * this.radius ** 2; }
}
class Rectangle implements Shape {
    getArea() { return this.width * this.height; }
}
```

## Replace Inheritance with Composition

**When**: Subclass only uses a fraction of parent, or "is-a" relationship is forced.

```python
# Before
class AudioPlayer(MediaWidget):  # inherits 50 methods, uses 5
    pass

# After
class AudioPlayer:
    def __init__(self):
        self.media = MediaWidget()  # delegate what you need
```

## Introduce Parameter Object

**When**: 3+ parameters travel together across multiple functions.

```go
// Before
func createUser(name string, email string, age int, role string, dept string) {}

// After
type CreateUserParams struct {
    Name  string
    Email string
    Age   int
    Role  string
    Dept  string
}
func createUser(params CreateUserParams) {}
```

## Replace Magic Values with Constants

```python
# Before
if response.status_code == 429:
    time.sleep(60)

# After
RATE_LIMIT_STATUS = 429
RATE_LIMIT_COOLDOWN_SECONDS = 60
if response.status_code == RATE_LIMIT_STATUS:
    time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
```

## Strangler Fig — Detailed Example

```python
# Phase 1: Facade over legacy
class PaymentFacade:
    def process_payment(self, order):
        return self.legacy_processor.doPayment(order.to_legacy())

# Phase 2: New service alongside
class PaymentService:
    def process_payment(self, order): ...

# Phase 3: Feature-flagged migration
class PaymentFacade:
    def process_payment(self, order):
        if feature_flag("use_new_payment"):
            return self.new_service.process_payment(order)
        return self.legacy.doPayment(order.to_legacy())
```

## Parallel Implementation

```python
def process(data):
    old_result = old_implementation(data)
    new_result = new_implementation(data)
    if old_result != new_result:
        log.warning(f"Mismatch: {old_result} vs {new_result}")
    return old_result  # switch to new_result when confident
```

## Metrics Dashboard Template

```yaml
cyclomatic_complexity: { current: 15.2, target: 10.0 }
code_duplication: { current: 23%, target: 5% }
test_coverage: { unit: 45%, integration: 12%, target: 80%/60% }
dependency_health: { outdated_major: 12, security_vulns: 7 }
```

## Impact Assessment Example

```
Debt Item: Duplicate user validation logic (5 files)
Time Impact: 2 hrs/bug fix, 4 hrs/feature change
Monthly: ~20 hours | Annual: 240 hrs x $150/hr = $36,000
```
