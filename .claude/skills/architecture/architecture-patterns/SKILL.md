---
name: architecture-patterns
description: Implement proven backend architecture patterns including Clean Architecture, Hexagonal Architecture, and Domain-Driven Design. Use when architecting complex backend systems or refactoring existing applications for better maintainability.
---

# Architecture Patterns

## Clean Architecture

### Directory Structure
```
app/
├── domain/           # Entities & business rules
│   ├── entities/
│   ├── value_objects/
│   └── interfaces/   # Abstract interfaces (ports)
├── use_cases/        # Application business rules
├── adapters/         # Interface implementations
│   ├── repositories/
│   ├── controllers/
│   └── gateways/
└── infrastructure/   # Framework & external concerns
```

### Implementation

```python
# domain/entities/user.py
@dataclass
class User:
    id: str
    email: str
    name: str
    created_at: datetime
    is_active: bool = True

    def deactivate(self):
        self.is_active = False

    def can_place_order(self) -> bool:
        return self.is_active

# domain/interfaces/user_repository.py
class IUserRepository(ABC):
    @abstractmethod
    async def find_by_id(self, user_id: str) -> Optional[User]: pass
    @abstractmethod
    async def find_by_email(self, email: str) -> Optional[User]: pass
    @abstractmethod
    async def save(self, user: User) -> User: pass
    @abstractmethod
    async def delete(self, user_id: str) -> bool: pass

# use_cases/create_user.py
@dataclass
class CreateUserRequest:
    email: str
    name: str

class CreateUserUseCase:
    def __init__(self, user_repository: IUserRepository):
        self.user_repository = user_repository

    async def execute(self, request: CreateUserRequest) -> CreateUserResponse:
        existing = await self.user_repository.find_by_email(request.email)
        if existing:
            return CreateUserResponse(user=None, success=False, error="Email already exists")
        user = User(id=str(uuid.uuid4()), email=request.email,
                    name=request.name, created_at=datetime.now())
        saved_user = await self.user_repository.save(user)
        return CreateUserResponse(user=saved_user, success=True)

# adapters/repositories/postgres_user_repository.py
class PostgresUserRepository(IUserRepository):
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def find_by_id(self, user_id: str) -> Optional[User]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
            return self._to_entity(row) if row else None

    async def save(self, user: User) -> User:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO users (id, email, name, created_at, is_active)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET email = $2, name = $3, is_active = $5
            """, user.id, user.email, user.name, user.created_at, user.is_active)
            return user

# adapters/controllers/user_controller.py
@router.post("/users")
async def create_user(dto: CreateUserDTO,
                      use_case: CreateUserUseCase = Depends(get_create_user_use_case)):
    request = CreateUserRequest(email=dto.email, name=dto.name)
    response = await use_case.execute(request)
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return {"user": response.user}
```

## Hexagonal Architecture (Ports and Adapters)

```python
# Core domain (hexagon center)
class OrderService:
    def __init__(self, order_repository: OrderRepositoryPort,
                 payment_gateway: PaymentGatewayPort,
                 notification_service: NotificationPort):
        self.orders = order_repository
        self.payments = payment_gateway
        self.notifications = notification_service

    async def place_order(self, order: Order) -> OrderResult:
        if not order.is_valid():
            return OrderResult(success=False, error="Invalid order")
        payment = await self.payments.charge(amount=order.total, customer=order.customer_id)
        if not payment.success:
            return OrderResult(success=False, error="Payment failed")
        order.mark_as_paid()
        saved_order = await self.orders.save(order)
        await self.notifications.send(to=order.customer_email,
                                       subject="Order confirmed",
                                       body=f"Order {order.id} confirmed")
        return OrderResult(success=True, order=saved_order)

# Ports (interfaces)
class OrderRepositoryPort(ABC):
    @abstractmethod
    async def save(self, order: Order) -> Order: pass

class PaymentGatewayPort(ABC):
    @abstractmethod
    async def charge(self, amount: Money, customer: str) -> PaymentResult: pass

# Adapters
class StripePaymentAdapter(PaymentGatewayPort):
    async def charge(self, amount: Money, customer: str) -> PaymentResult:
        try:
            charge = self.stripe.Charge.create(amount=amount.cents, currency=amount.currency, customer=customer)
            return PaymentResult(success=True, transaction_id=charge.id)
        except stripe.error.CardError as e:
            return PaymentResult(success=False, error=str(e))

class MockPaymentAdapter(PaymentGatewayPort):
    async def charge(self, amount: Money, customer: str) -> PaymentResult:
        return PaymentResult(success=True, transaction_id="mock-123")
```

## Domain-Driven Design

```python
# Value Objects (immutable)
@dataclass(frozen=True)
class Email:
    value: str
    def __post_init__(self):
        if "@" not in self.value:
            raise ValueError("Invalid email")

@dataclass(frozen=True)
class Money:
    amount: int  # cents
    currency: str
    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Currency mismatch")
        return Money(self.amount + other.amount, self.currency)

# Entity (with identity, mutable state)
class Order:
    def __init__(self, id: str, customer: Customer):
        self.id = id
        self.customer = customer
        self.items: List[OrderItem] = []
        self.status = OrderStatus.PENDING
        self._events: List[DomainEvent] = []

    def add_item(self, product: Product, quantity: int):
        item = OrderItem(product, quantity)
        self.items.append(item)
        self._events.append(ItemAddedEvent(self.id, item))

    def submit(self):
        if not self.items:
            raise ValueError("Cannot submit empty order")
        if self.status != OrderStatus.PENDING:
            raise ValueError("Order already submitted")
        self.status = OrderStatus.SUBMITTED
        self._events.append(OrderSubmittedEvent(self.id))

# Aggregate root
class Customer:
    def __init__(self, id: str, email: Email):
        self.id = id
        self.email = email
        self._addresses: List[Address] = []

    def add_address(self, address: Address):
        if len(self._addresses) >= 5:
            raise ValueError("Maximum 5 addresses allowed")
        self._addresses.append(address)

# Repository
class OrderRepository:
    async def save(self, order: Order):
        await self._persist(order)
        await self._publish_events(order._events)
        order._events.clear()
```

## Key Principles

1. **Dependency Rule**: Dependencies always point inward
2. **Interface Segregation**: Small, focused interfaces
3. **Business Logic in Domain**: Keep frameworks out of core
4. **Test Independence**: Core testable without infrastructure
5. **Rich Domain Models**: Behavior with data, not anemic entities

## Pitfalls

- **Anemic Domain**: Entities with only data, no behavior
- **Framework Coupling**: Business logic depends on frameworks
- **Fat Controllers**: Business logic in controllers
- **Repository Leakage**: Exposing ORM objects
- **Over-Engineering**: Clean architecture for simple CRUD
