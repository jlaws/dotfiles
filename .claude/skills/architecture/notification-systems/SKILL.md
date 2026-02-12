---
name: notification-systems
description: Build multi-channel notification systems with email, push, in-app, and webhook delivery
---

# Notification Systems

## Decision Table

| Urgency | Type | Channel | Fallback |
|---------|------|---------|----------|
| Critical (outage, security) | Transactional | Push + SMS + Email | All channels simultaneously |
| High (payment, auth) | Transactional | Push + Email | Email if push undelivered after 5min |
| Medium (updates, activity) | Transactional | Push or In-app | Email digest if unread after 24h |
| Low (marketing, tips) | Bulk | Email | None |
| System-to-system | Webhook | HTTP POST | Retry with exponential backoff |

## Email Sending

### SES with boto3

```python
# email_sender.py
import boto3
from botocore.exceptions import ClientError

class SESEmailSender:
    def __init__(self, region="us-east-1"):
        self.client = boto3.client("ses", region_name=region)

    def send(self, to: str, subject: str, html_body: str,
             from_addr: str = "noreply@example.com") -> str:
        try:
            resp = self.client.send_email(
                Source=from_addr,
                Destination={"ToAddresses": [to]},
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": {"Html": {"Data": html_body, "Charset": "UTF-8"}},
                },
            )
            return resp["MessageId"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "MessageRejected":
                raise ValueError(f"Rejected: {e}") from e
            raise
```

## Push Notifications (FCM HTTP v1)

```python
# push_sender.py
import google.auth.transport.requests
from google.oauth2 import service_account
import httpx

class FCMSender:
    FCM_URL = "https://fcm.googleapis.com/v1/projects/{}/messages:send"

    def __init__(self, project_id: str, sa_path: str):
        self.url = self.FCM_URL.format(project_id)
        creds = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/firebase.messaging"],
        )
        creds.refresh(google.auth.transport.requests.Request())
        self.headers = {"Authorization": f"Bearer {creds.token}"}

    def send_to_device(self, token: str, title: str, body: str,
                        data: dict | None = None) -> dict:
        payload = {"message": {
            "token": token,
            "notification": {"title": title, "body": body},
            "data": data or {},
            "android": {"priority": "high"},
            "apns": {"headers": {"apns-priority": "10"}},
        }}
        resp = httpx.post(self.url, json=payload, headers=self.headers)
        resp.raise_for_status()
        return resp.json()
```

## In-App Notification System

### Database Schema

```sql
CREATE TABLE notifications (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id),
    type        VARCHAR(50) NOT NULL,
    title       VARCHAR(255) NOT NULL,
    body        TEXT,
    data        JSONB DEFAULT '{}',
    read_at     TIMESTAMPTZ,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_notif_user_unread
    ON notifications(user_id, created_at DESC) WHERE read_at IS NULL;
```

### API Layer

```python
# notification_service.py
from uuid import UUID
import asyncpg

class InAppNotificationService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def create(self, user_id: UUID, type_: str,
                     title: str, body: str = "", data: dict = None) -> UUID:
        row = await self.pool.fetchrow("""
            INSERT INTO notifications (user_id, type, title, body, data)
            VALUES ($1, $2, $3, $4, $5::jsonb) RETURNING id
        """, user_id, type_, title, body, data or {})
        return row["id"]

    async def get_unread(self, user_id: UUID, limit: int = 50) -> list:
        return await self.pool.fetch("""
            SELECT id, type, title, body, data, created_at
            FROM notifications
            WHERE user_id = $1 AND read_at IS NULL
            ORDER BY created_at DESC LIMIT $2
        """, user_id, limit)

    async def mark_read(self, user_id: UUID, ids: list[UUID]):
        await self.pool.execute("""
            UPDATE notifications SET read_at = now()
            WHERE user_id = $1 AND id = ANY($2) AND read_at IS NULL
        """, user_id, ids)
```

## Webhook Delivery

```python
# webhook_delivery.py
import hashlib, hmac, time, asyncio
import httpx

class WebhookDelivery:
    MAX_RETRIES = 5

    def __init__(self, signing_secret: str):
        self.secret = signing_secret.encode()

    def sign_payload(self, payload: bytes, ts: int) -> str:
        return hmac.new(self.secret, f"{ts}.{payload.decode()}".encode(),
                        hashlib.sha256).hexdigest()

    async def deliver(self, url: str, payload: dict) -> bool:
        """Deliver with retry and exponential backoff."""
        body, ts = httpx.json_serialize(payload), int(time.time())
        headers = {"Content-Type": "application/json",
                   "X-Webhook-Signature": self.sign_payload(body, ts),
                   "X-Webhook-Timestamp": str(ts)}
        async with httpx.AsyncClient(timeout=10) as client:
            for attempt in range(self.MAX_RETRIES):
                try:
                    resp = await client.post(url, content=body, headers=headers)
                    if resp.status_code < 300:
                        return True
                    if resp.status_code < 500:  # client error, don't retry
                        return False
                except httpx.RequestError:
                    pass
                await asyncio.sleep(2 ** attempt)  # 1, 2, 4, 8, 16s
        return False
```

## Notification Router

```python
# router.py
from dataclasses import dataclass
from enum import Enum

class Channel(Enum):
    EMAIL = "email"
    PUSH = "push"
    IN_APP = "in_app"
    SMS = "sms"

@dataclass
class UserPreferences:
    enabled_channels: set[Channel]
    quiet_hours: tuple[int, int] | None  # (start_hour, end_hour) UTC

class NotificationRouter:
    ROUTING = {
        "security_alert":  [Channel.PUSH, Channel.SMS, Channel.EMAIL],
        "payment_receipt":  [Channel.EMAIL, Channel.PUSH],
        "comment_reply":    [Channel.PUSH, Channel.IN_APP],
        "marketing":        [Channel.EMAIL],
    }

    def resolve_channels(self, notif_type: str, prefs: UserPreferences) -> list[Channel]:
        return [ch for ch in self.ROUTING.get(notif_type, [Channel.IN_APP])
                if ch in prefs.enabled_channels]
```

## Template Rendering and Rate Limiting

```python
# templates.py
from jinja2 import Environment, FileSystemLoader, select_autoescape

def render_notification(template_dir: str, template_name: str,
                         context: dict, locale: str = "en") -> tuple[str, str]:
    """Return (subject, html_body) for a notification template."""
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html"]),
    )
    html = env.get_template(f"{locale}/{template_name}.html").render(**context)
    subject = env.get_template(
        f"{locale}/{template_name}.subject.txt").render(**context).strip()
    return subject, html
```

```python
# rate_limiter.py
import redis.asyncio as redis

class NotificationRateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.r = redis_client

    async def check_and_increment(self, user_id: str, channel: str,
                                    max_per_hour: int = 10) -> bool:
        """Return True if under limit, False if rate-limited."""
        key = f"notif_rate:{user_id}:{channel}"
        pipe = self.r.pipeline()
        pipe.incr(key)
        pipe.expire(key, 3600)
        return (await pipe.execute())[0] <= max_per_hour
```

## Gotchas

- **SES sandbox mode** -- new accounts only send to verified addresses; request production access first
- **FCM token rotation** -- device tokens expire; handle `UNREGISTERED` errors by removing stale tokens
- **Webhook replay attacks** -- validate timestamp is within 5 minutes alongside signature
- **Email deliverability** -- set up SPF, DKIM, and DMARC records or emails land in spam
- **In-app notification bloat** -- add TTL or archival; millions of read notifications slow queries
- **Provider rate limits** -- email providers enforce own limits (SES: 14/sec); push: FCM/APNs 4KB max payload
- **Quiet hours timezone** -- check user timezone, not server; store as UTC offset
- **Webhook idempotency** -- receivers may get duplicates; include idempotency key in headers
