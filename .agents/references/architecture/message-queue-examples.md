# Message Queue Code Examples

## Kafka Producer/Consumer

```python
from confluent_kafka import Producer, Consumer, KafkaError
import json

def create_kafka_producer(bootstrap_servers: str) -> Producer:
    return Producer({
        "bootstrap.servers": bootstrap_servers,
        "acks": "all",
        "enable.idempotence": True,
        "max.in.flight.requests.per.connection": 5,
        "delivery.timeout.ms": 120000,
        "linger.ms": 5,
        "compression.type": "lz4",
    })

def publish_event(producer: Producer, topic: str, key: str, event: dict):
    """Publish with partition key for ordering guarantees."""
    producer.produce(
        topic=topic,
        key=key.encode("utf-8"),
        value=json.dumps(event).encode("utf-8"),
        callback=lambda err, msg: err and print(f"Delivery failed: {err}"),
    )
    producer.poll(0)

def consume_loop(bootstrap_servers: str, group_id: str, topics: list[str], handler):
    """Process-then-commit loop with graceful shutdown."""
    consumer = Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
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
    channel.exchange_declare(exchange="events.fanout", exchange_type="fanout", durable=True)
    channel.exchange_declare(exchange="events.topic", exchange_type="topic", durable=True)
    channel.exchange_declare(exchange="events.direct", exchange_type="direct", durable=True)

def publish_with_confirms(channel: pika.channel.Channel, exchange: str,
                          routing_key: str, body: dict):
    """Publish with publisher confirms for reliability."""
    channel.confirm_delivery()
    channel.basic_publish(
        exchange=exchange, routing_key=routing_key, body=json.dumps(body),
        properties=pika.BasicProperties(
            delivery_mode=2,
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
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    channel.basic_consume(queue=queue, on_message_callback=callback)
    channel.start_consuming()
```

## Celery Task Queues with Retry

```python
from celery import Celery

app = Celery("tasks", broker="redis://localhost:6379/0")
app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    task_serializer="json", result_serializer="json", accept_content=["json"],
)

@app.task(
    bind=True, max_retries=5, default_retry_delay=60,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True, retry_backoff_max=600,
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
            WaitTimeSeconds=20,
            VisibilityTimeout=300,
        )
        for message in response.get("Messages", []):
            try:
                handler(json.loads(message["Body"]))
                sqs.delete_message(QueueUrl=queue_url,
                                   ReceiptHandle=message["ReceiptHandle"])
            except Exception as e:
                print(f"Failed: {e}")

# DLQ setup: attach redrive policy to main queue
sqs.set_queue_attributes(
    QueueUrl=main_queue_url,
    Attributes={"RedrivePolicy": json.dumps({
        "deadLetterTargetArn": dlq_arn,
        "maxReceiveCount": "3",
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

## Event Bus and Idempotency

```python
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import uuid, hashlib

@dataclass
class DomainEvent:
    event_type: str
    aggregate_id: str
    data: dict
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = 1
    correlation_id: str = ""
    idempotency_key: str = ""

class EventBus:
    """Lightweight in-process event bus with handler registry."""
    def __init__(self):
        self._handlers: dict[str, list] = {}

    def subscribe(self, event_type: str, handler):
        self._handlers.setdefault(event_type, []).append(handler)

    async def publish(self, event: DomainEvent):
        for handler in self._handlers.get(event.event_type, []):
            await handler(asdict(event))

class IdempotencyStore:
    """Redis-backed idempotency check with TTL."""
    def __init__(self, redis_client, ttl_seconds: int = 86400):
        self.redis = redis_client
        self.ttl = ttl_seconds

    def check_and_set(self, key: str) -> bool:
        """Returns True if already processed (duplicate)."""
        result = self.redis.set(f"idempotency:{key}", "1", nx=True, ex=self.ttl)
        return result is None
```
