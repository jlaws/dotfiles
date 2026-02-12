---
name: message-queue-patterns
description: Design reliable message-driven architectures with queues, topics, and task workers
---

# Message Queue Patterns

## Queue Technology Selection

| Technology | Throughput | Latency | Ordering | Best For |
|------------|-----------|---------|----------|----------|
| **Kafka** | Millions/s | 5-15ms | Per-partition | Event streaming, log aggregation, replay |
| **RabbitMQ** | 50K/s | <1ms | Per-queue | Task routing, RPC, complex routing rules |
| **Celery** | 10K/s | 5-50ms | None (FIFO optional) | Python task queues, scheduled jobs |
| **SQS** | Unlimited | 20-50ms | FIFO optional | Serverless, AWS-native, zero ops |
| **Redis Streams** | 100K/s | <1ms | Per-stream | Lightweight streaming, ephemeral data |

**Rule of thumb:** Kafka for event streaming, RabbitMQ for task routing, SQS for serverless glue, Celery for Python-native workers.

## Kafka Producer/Consumer

```python
from confluent_kafka import Producer, Consumer, KafkaError
import json

def create_kafka_producer(bootstrap_servers: str) -> Producer:
    return Producer({
        "bootstrap.servers": bootstrap_servers,
        "acks": "all",                    # Wait for all replicas
        "enable.idempotence": True,       # Exactly-once per partition
        "max.in.flight.requests.per.connection": 5,
        "delivery.timeout.ms": 120000,    # 2min total delivery timeout
        "linger.ms": 5,                   # Batch for 5ms before sending
        "compression.type": "lz4",
    })

def publish_event(producer: Producer, topic: str, key: str, event: dict):
    """Publish with partition key for ordering guarantees."""
    producer.produce(
        topic=topic,
        key=key.encode("utf-8"),          # Same key -> same partition -> ordered
        value=json.dumps(event).encode("utf-8"),
        callback=lambda err, msg: err and print(f"Delivery failed: {err}"),
    )
    producer.poll(0)                      # Trigger callbacks without blocking

def consume_loop(bootstrap_servers: str, group_id: str, topics: list[str], handler):
    """Process-then-commit loop with graceful shutdown."""
    consumer = Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,      # Manual commit after processing
        "max.poll.interval.ms": 300000,
    })
    consumer.subscribe(topics)
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise RuntimeError(msg.error())
            handler(json.loads(msg.value().decode("utf-8")))
            consumer.commit(asynchronous=False)
    finally:
        consumer.close()
```

## RabbitMQ Exchange Patterns

```python
import pika, uuid, json

def setup_exchanges(channel: pika.channel.Channel):
    # Fanout: broadcast to all bound queues (notifications, cache invalidation)
    channel.exchange_declare(exchange="events.fanout", exchange_type="fanout", durable=True)
    # Topic: route by pattern (order.created, order.shipped -> order.*)
    channel.exchange_declare(exchange="events.topic", exchange_type="topic", durable=True)
    # Direct: route by exact key (payment.process -> payment worker)
    channel.exchange_declare(exchange="events.direct", exchange_type="direct", durable=True)

def publish_with_confirms(channel: pika.channel.Channel, exchange: str,
                          routing_key: str, body: dict):
    """Publish with publisher confirms for reliability."""
    channel.confirm_delivery()
    channel.basic_publish(
        exchange=exchange, routing_key=routing_key, body=json.dumps(body),
        properties=pika.BasicProperties(
            delivery_mode=2,              # Persistent message
            content_type="application/json",
            message_id=str(uuid.uuid4()),
        ),
    )

def consume_with_ack(channel: pika.channel.Channel, queue: str, handler):
    """Manual ack after successful processing."""
    channel.basic_qos(prefetch_count=10)
    def callback(ch, method, properties, body):
        try:
            handler(json.loads(body))
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)  # -> DLQ
    channel.basic_consume(queue=queue, on_message_callback=callback)
    channel.start_consuming()
```

## Celery Task Queues with Retry

```python
from celery import Celery

app = Celery("tasks", broker="redis://localhost:6379/0")
app.conf.update(
    task_acks_late=True,                  # Ack after completion (not receipt)
    worker_prefetch_multiplier=1,         # One task at a time per worker
    task_reject_on_worker_lost=True,      # Requeue if worker dies
    task_serializer="json", result_serializer="json", accept_content=["json"],
)

@app.task(
    bind=True, max_retries=5, default_retry_delay=60,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,                   # Exponential backoff
    retry_backoff_max=600,                # Cap at 10min
    retry_jitter=True, acks_late=True,
)
def process_order(self, order_id: str, idempotency_key: str):
    """Idempotent task with exponential backoff retry."""
    if already_processed(idempotency_key):
        return {"status": "duplicate"}
    try:
        result = do_order_processing(order_id)
        mark_processed(idempotency_key)
        return result
    except Exception as exc:
        raise self.retry(exc=exc)
```

## SQS Polling and Dead Letter Queue

```python
import boto3, json

sqs = boto3.client("sqs")

def poll_sqs(queue_url: str, handler, max_messages: int = 10):
    """Long-poll SQS with visibility timeout management."""
    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=20,           # Long polling (reduces API calls)
            VisibilityTimeout=300,        # 5min to process before redelivery
        )
        for message in response.get("Messages", []):
            try:
                handler(json.loads(message["Body"]))
                sqs.delete_message(QueueUrl=queue_url,
                                   ReceiptHandle=message["ReceiptHandle"])
            except Exception as e:
                print(f"Failed: {e}")     # Returns after visibility timeout

# DLQ setup: attach redrive policy to main queue
sqs.set_queue_attributes(
    QueueUrl=main_queue_url,
    Attributes={"RedrivePolicy": json.dumps({
        "deadLetterTargetArn": dlq_arn,
        "maxReceiveCount": "3",           # Move to DLQ after 3 failures
    })},
)

def replay_dlq(dlq_url: str, main_queue_url: str):
    """Selectively replay DLQ messages after root cause fix."""
    response = sqs.receive_message(QueueUrl=dlq_url, MaxNumberOfMessages=10)
    for msg in response.get("Messages", []):
        body = json.loads(msg["Body"])
        if is_retriable(body):
            sqs.send_message(QueueUrl=main_queue_url, MessageBody=msg["Body"])
        sqs.delete_message(QueueUrl=dlq_url, ReceiptHandle=msg["ReceiptHandle"])
```

## Event Schema and Bus

```python
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid

@dataclass
class DomainEvent:
    event_type: str
    aggregate_id: str
    data: dict
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    schema_version: int = 1               # Version from day one
    correlation_id: str = ""              # Trace across services
    idempotency_key: str = ""             # Dedup key

class EventBus:
    """Lightweight in-process event bus with handler registry."""
    def __init__(self):
        self._handlers: dict[str, list] = {}

    def subscribe(self, event_type: str, handler):
        self._handlers.setdefault(event_type, []).append(handler)

    async def publish(self, event: DomainEvent):
        for handler in self._handlers.get(event.event_type, []):
            await handler(asdict(event))
```

## Idempotency Key Pattern

```python
import hashlib

def generate_idempotency_key(entity_id: str, action: str, timestamp: str) -> str:
    """Deterministic key from business identifiers."""
    return hashlib.sha256(f"{entity_id}:{action}:{timestamp}".encode()).hexdigest()

class IdempotencyStore:
    """Redis-backed idempotency check with TTL."""
    def __init__(self, redis_client, ttl_seconds: int = 86400):
        self.redis = redis_client
        self.ttl = ttl_seconds

    def check_and_set(self, key: str) -> bool:
        """Returns True if already processed (duplicate)."""
        result = self.redis.set(f"idempotency:{key}", "1", nx=True, ex=self.ttl)
        return result is None             # None = key already existed
```

## Gotchas

- **Ordering across partitions**: Kafka only orders within a partition; use consistent partition keys
- **Poison messages**: Always configure DLQ; one bad message can block an entire queue
- **At-least-once is the default**: Design every consumer to be idempotent
- **Celery visibility timeout**: If task takes longer than visibility timeout, it gets redelivered
- **RabbitMQ queue depth**: Unbounded queues cause memory pressure; set `x-max-length`
- **SQS FIFO throughput**: 300 msg/s per group ID, 3000/s per queue; plan group IDs carefully
- **Consumer lag monitoring**: Alert on growing lag before it becomes an outage
- **Message size limits**: SQS 256KB, Kafka default 1MB, RabbitMQ no hard limit but >128KB hurts
- **Broker is not a database**: Don't use queues for long-term storage; process and persist
