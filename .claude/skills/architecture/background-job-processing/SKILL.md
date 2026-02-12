---
name: background-job-processing
description: Design reliable background job systems with retry strategies, idempotency, and scheduling
---

# Background Job Processing

## Decision Table

| Framework | Language | Scale | Priority Queues | Scheduling | Best For |
|-----------|---------|-------|----------------|------------|----------|
| **Celery** | Python | High | Yes (via queues) | Celery Beat | Django/Flask, ML pipelines |
| **BullMQ** | Node.js | High | Native | Built-in | TypeScript APIs, sandboxed workers |
| **Sidekiq** | Ruby | High | Native | sidekiq-cron | Rails applications |
| **RQ** | Python | Low-Med | Basic | rq-scheduler | Simple Python apps |
| **Dramatiq** | Python | Med-High | Yes | APScheduler | Celery alternative, simpler API |
| **Temporal** | Any | Very High | Via workflows | Built-in | Complex workflows, saga patterns |

## Celery Task Patterns

### Retry with Exponential Backoff

```python
from celery import Celery, Task
import requests

app = Celery("tasks", broker="redis://localhost:6379/0",
             backend="redis://localhost:6379/1")

class BaseTaskWithRetry(Task):
    """Base task class with exponential backoff defaults."""
    autoretry_for = (requests.RequestException, ConnectionError)
    retry_backoff = True           # exponential backoff
    retry_backoff_max = 600        # cap at 10 minutes
    retry_jitter = True            # randomize to prevent thundering herd
    max_retries = 5
    acks_late = True               # ack after completion, not receipt
    reject_on_worker_lost = True   # requeue if worker crashes mid-task

@app.task(base=BaseTaskWithRetry, bind=True)
def send_webhook(self, url: str, payload: dict):
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return {"status": resp.status_code, "url": url}

@app.task(bind=True, max_retries=3)
def process_payment(self, order_id: str, amount: float):
    """Manual retry with custom backoff intervals."""
    try:
        return payment_gateway.charge(order_id, amount)
    except PaymentGatewayTimeout as exc:
        countdown = [10, 60, 300][self.request.retries]  # 10s, 60s, 300s
        raise self.retry(exc=exc, countdown=countdown)
    except PaymentDeclined:
        return {"status": "declined", "order_id": order_id}  # no retry
```

### Chain, Chord, and Group

```python
from celery import chain, chord, group

# Chain: sequential pipeline
pipeline = chain(download_file.s(url), parse_csv.s(), store_results.s())

# Group: parallel fan-out
batch = group(process_image.s(img_id) for img_id in image_ids)

# Chord: parallel + callback when all complete
workflow = chord([analyze_chunk.s(c) for c in chunks], aggregate_results.s())

# Nested: fan-out -> aggregate -> notify
full = chain(
    chord([process_item.s(i) for i in items], merge_results.s()),
    send_notification.s(user_email="admin@example.com"),
)
```

### Rate Limiting

```python
@app.task(rate_limit="10/m")  # 10 per minute per worker
def call_external_api(endpoint: str, params: dict):
    return requests.get(endpoint, params=params).json()
```

## BullMQ Patterns (Node.js)

```python
# BullMQ TypeScript reference:
"""
import { Queue, Worker } from 'bullmq';
const queue = new Queue('email', { connection: { host: 'localhost' } });

await queue.add('send', { to: 'user@example.com' }, {
  priority: 1,                                // lower = higher priority
  attempts: 3,
  backoff: { type: 'exponential', delay: 1000 },
  removeOnComplete: { age: 86400 },           // cleanup after 24h
});

const worker = new Worker('email', './processors/email.js', {
  concurrency: 5,
  limiter: { max: 100, duration: 60000 },     // 100/min rate limit
  sandboxedProcessors: true,                  // crash isolation
});
"""
```

## Job Idempotency

```python
import hashlib, redis, json

redis_client = redis.Redis()

def make_idempotency_key(task_name: str, args: tuple, kwargs: dict) -> str:
    payload = json.dumps({"task": task_name, "args": args, "kwargs": kwargs},
                         sort_keys=True)
    return f"idem:{hashlib.sha256(payload.encode()).hexdigest()}"

@app.task(bind=True)
def idempotent_charge(self, order_id: str, amount: float):
    """Exactly-once processing via idempotency key."""
    key = make_idempotency_key("charge", (order_id,), {"amount": amount})
    existing = redis_client.get(key)
    if existing:
        return json.loads(existing)
    result = payment_gateway.charge(order_id, amount)
    redis_client.setex(key, 86400, json.dumps(result))  # 24h TTL
    return result
```

### Deduplication with Redis Locks

```python
from contextlib import contextmanager

@contextmanager
def distributed_lock(key: str, timeout: int = 300):
    """Prevent duplicate job execution with Redis lock."""
    lock_key = f"lock:{key}"
    acquired = redis_client.set(lock_key, "1", nx=True, ex=timeout)
    if not acquired:
        raise JobAlreadyRunning(f"Lock {key} held by another worker")
    try:
        yield
    finally:
        redis_client.delete(lock_key)

@app.task(bind=True)
def sync_user_data(self, user_id: str):
    try:
        with distributed_lock(f"sync:{user_id}", timeout=600):
            fetch_from_external_api(user_id)
            update_local_database(user_id)
    except JobAlreadyRunning:
        return {"status": "skipped", "reason": "already_running"}
```

## Dead Letter Queue Handling

```python
from celery.signals import task_failure
from datetime import datetime

@task_failure.connect
def on_task_failure(sender=None, task_id=None, exception=None,
                    args=None, kwargs=None, **kw):
    """Route permanently failed tasks to DLQ."""
    if sender.request.retries >= sender.max_retries:
        redis_client.lpush("dlq", json.dumps({
            "task_name": sender.name, "task_id": task_id,
            "args": args, "kwargs": kwargs,
            "exception": str(exception), "failed_at": datetime.utcnow().isoformat(),
        }))

def replay_dead_letters(limit=100):
    """Re-enqueue failed tasks from DLQ."""
    for _ in range(limit):
        entry = redis_client.rpop("dlq")
        if not entry:
            break
        data = json.loads(entry)
        app.send_task(data["task_name"], args=data["args"], kwargs=data["kwargs"])
```

## Cron-Style Scheduling

```python
from celery.schedules import crontab

app.conf.beat_schedule = {
    "cleanup-sessions": {
        "task": "tasks.cleanup_sessions",
        "schedule": crontab(minute=0, hour="*/6"),     # every 6h
    },
    "daily-report": {
        "task": "tasks.generate_report",
        "schedule": crontab(minute=0, hour=8),          # daily 8am
        "args": ("daily",),
    },
    "sync-inventory": {
        "task": "tasks.sync_inventory",
        "schedule": 300.0,                               # every 5min
        "options": {"expires": 280},                     # expire before next
    },
}
```

### APScheduler (Framework-Agnostic)

```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.redis import RedisJobStore

scheduler = BackgroundScheduler(
    jobstores={"default": RedisJobStore(host="localhost")},
    job_defaults={"coalesce": True, "max_instances": 1})
scheduler.add_job(cleanup, trigger=CronTrigger(hour="*/6"),
                  id="cleanup", replace_existing=True, misfire_grace_time=300)
```

## Priority Queue Implementation

```python
@app.task(queue="critical")
def process_refund(order_id: str): pass  # high priority

@app.task(queue="default")
def send_email(user_id: str): pass       # normal

@app.task(queue="bulk")
def generate_report(year: int): pass     # low priority

# celery -A tasks worker -Q critical,default,bulk -c 4
# Or dedicated workers: celery -A tasks worker -Q critical -c 8
```

## Gotchas

- **Late acks without idempotency**: `acks_late=True` requeues on crash = double execution; always pair with idempotency
- **Celery chord fragility**: If any header task fails, callback never fires; add error handling
- **Redis broker loses messages**: Redis not durable by default; use RabbitMQ for guaranteed delivery
- **Beat schedule drift**: Multiple Beat instances = duplicate jobs; run exactly one Beat process
- **Task serialization traps**: Pickle is insecure; use JSON serializer, pass IDs not objects
- **Worker memory leaks**: Set `--max-tasks-per-child=1000` to recycle long-running workers
- **Rate limits are per-worker**: `rate_limit="10/m"` is per process, not global; use Redis token bucket
- **Invisible timeout**: `time_limit` kills without retry; use `soft_time_limit` + exception handler
