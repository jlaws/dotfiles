---
name: microservices-patterns
description: Design microservices architectures with service boundaries, event-driven communication, and resilience patterns. Use when building distributed systems, decomposing monoliths, or implementing microservices.
---

# Microservices Patterns

## Service Decomposition

**By Business Capability**: OrderService, PaymentService, InventoryService
**By Subdomain (DDD)**: Bounded contexts map to services
**Strangler Fig**: Gradually extract from monolith, proxy routes to old/new

## Communication Patterns

| Pattern | Examples | When |
|---------|----------|------|
| **Synchronous** | REST, gRPC, GraphQL | Request/response needed |
| **Asynchronous** | Kafka, RabbitMQ, SQS | Event-driven, decoupled |

## Service Decomposition by Business Capability

```python
class OrderService:
    async def create_order(self, order_data):
        order = Order.create(order_data)
        await self.event_bus.publish(OrderCreatedEvent(
            order_id=order.id, customer_id=order.customer_id,
            items=order.items, total=order.total))
        return order

class PaymentService:
    async def process_payment(self, payment_request):
        result = await self.payment_gateway.charge(
            amount=payment_request.amount, customer=payment_request.customer_id)
        if result.success:
            await self.event_bus.publish(PaymentCompletedEvent(
                order_id=payment_request.order_id, transaction_id=result.transaction_id))
        return result
```

## API Gateway

```python
class APIGateway:
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=5.0)

    @circuit(failure_threshold=5, recovery_timeout=30)
    async def call_order_service(self, path, method="GET", **kwargs):
        response = await self.http_client.request(method, f"{self.order_service_url}{path}", **kwargs)
        response.raise_for_status()
        return response.json()

    async def create_order_aggregate(self, order_id):
        """Aggregate data from multiple services."""
        order, payment, inventory = await asyncio.gather(
            self.call_order_service(f"/orders/{order_id}"),
            self.call_payment_service(f"/payments/order/{order_id}"),
            self.call_inventory_service(f"/reservations/order/{order_id}"),
            return_exceptions=True)
        result = {"order": order}
        if not isinstance(payment, Exception): result["payment"] = payment
        if not isinstance(inventory, Exception): result["inventory"] = inventory
        return result
```

## Event-Driven Communication (Kafka)

```python
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

class EventBus:
    def __init__(self, bootstrap_servers):
        self.bootstrap_servers = bootstrap_servers

    async def publish(self, event: DomainEvent):
        await self.producer.send_and_wait(
            event.event_type, value=asdict(event), key=event.aggregate_id.encode())

    async def subscribe(self, topic, handler):
        consumer = AIOKafkaConsumer(
            topic, bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode()), group_id="my-service")
        await consumer.start()
        async for message in consumer:
            await handler(message.value)
```

## Saga Pattern (Distributed Transactions)

```python
class OrderFulfillmentSaga:
    def __init__(self):
        self.steps = [
            SagaStep("create_order", self.create_order, self.cancel_order),
            SagaStep("reserve_inventory", self.reserve_inventory, self.release_inventory),
            SagaStep("process_payment", self.process_payment, self.refund_payment),
            SagaStep("confirm_order", self.confirm_order, self.cancel_order_confirmation)
        ]

    async def execute(self, order_data):
        completed_steps = []
        context = {"order_data": order_data}
        try:
            for step in self.steps:
                result = await step.action(context)
                if not result.success:
                    await self.compensate(completed_steps, context)
                    return SagaResult(status=SagaStatus.FAILED, error=result.error)
                completed_steps.append(step)
                context.update(result.data)
            return SagaResult(status=SagaStatus.COMPLETED, data=context)
        except Exception as e:
            await self.compensate(completed_steps, context)
            return SagaResult(status=SagaStatus.FAILED, error=str(e))

    async def compensate(self, completed_steps, context):
        for step in reversed(completed_steps):
            try:
                await step.compensation(context)
            except Exception as e:
                print(f"Compensation failed for {step.name}: {e}")
```

## Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30, success_threshold=2):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at = None

    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED

    def _on_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = datetime.now()
```

## HTTP Client with Retries

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class ServiceClient:
    def __init__(self, base_url):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=2.0),
            limits=httpx.Limits(max_keepalive_connections=20))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get(self, path, **kwargs):
        response = await self.client.get(f"{self.base_url}{path}", **kwargs)
        response.raise_for_status()
        return response.json()
```

## Pitfalls

- **Distributed Monolith**: Tightly coupled services
- **Chatty Services**: Too many inter-service calls
- **Shared Databases**: Tight coupling through data
- **Synchronous Everything**: Tight coupling, poor resilience
- **No Compensation Logic**: Can't undo failed transactions
- **Premature Microservices**: Starting with microservices before understanding the domain
