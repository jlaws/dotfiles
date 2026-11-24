set -euo pipefail
mkdir -p app migrations
cat > app/__init__.py <<'PY'
PY
cat > app/db.py <<'PY'
"""Shared SQLite connection."""
import sqlite3

conn = sqlite3.connect("subscriptions.db")
PY
cat > app/celery_app.py <<'PY'
"""Celery application."""
from celery import Celery

celery = Celery("subscriptions", broker="redis://localhost:6379/0")
PY
cat > app/mail.py <<'PY'
"""Outbound email."""


def send(subscription_id, template):
    print(f"email {template} for subscription {subscription_id}")
PY
cat > app/billing.py <<'PY'
"""Payment provider client."""
import payments_sdk

client = payments_sdk.Client()


def charge(provider_customer_id, plan):
    return client.charges.create(customer=provider_customer_id, plan=plan)


def cancel_plan(provider_subscription_id):
    return client.subscriptions.cancel(provider_subscription_id)
PY
cat > app/routes.py <<'PY'
"""HTTP routes for subscriptions."""
from flask import Blueprint, jsonify

from app import services

bp = Blueprint("subscriptions", __name__)


@bp.post("/subscriptions/<int:subscription_id>/cancel")
def cancel(subscription_id):
    services.cancel_subscription(subscription_id)
    return jsonify({"status": "cancelled"})
PY
cat > app/services.py <<'PY'
"""Subscription business logic."""
from app.db import conn
from app.tasks import send_cancellation_email


def cancel_subscription(subscription_id):
    conn.execute("UPDATE subscriptions SET status = 'cancelled' WHERE id = ?", (subscription_id,))
    conn.commit()
    send_cancellation_email.delay(subscription_id)
PY
cat > app/tasks.py <<'PY'
"""Celery tasks."""
from app.celery_app import celery
from app.mail import send


@celery.task
def send_cancellation_email(subscription_id):
    send(subscription_id, template="cancelled")
PY
cat > app/scheduler.py <<'PY'
"""Periodic jobs (celery beat)."""
from celery.schedules import crontab

from app import billing
from app.celery_app import celery
from app.db import conn


@celery.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(crontab(hour=2, minute=0), renew_due_subscriptions.s())
    sender.add_periodic_task(crontab(minute=15), purge_expired_sessions.s())


@celery.task
def renew_due_subscriptions():
    rows = conn.execute(
        "SELECT id, provider_customer_id FROM subscriptions "
        "WHERE status = 'active' AND renews_at <= datetime('now')"
    ).fetchall()
    for subscription_id, customer in rows:
        billing.charge(customer, plan="monthly")
        conn.execute(
            "UPDATE subscriptions SET renews_at = datetime(renews_at, '+1 month') WHERE id = ?",
            (subscription_id,),
        )
        conn.commit()


@celery.task
def purge_expired_sessions():
    conn.execute("DELETE FROM sessions WHERE expires_at < datetime('now')")
    conn.commit()
PY
cat > app/billing_sync.py <<'PY'
"""Polls billing_events every minute and mirrors status changes to the payment provider."""
import time

from app import billing
from app.db import conn


def run_forever():
    while True:
        rows = conn.execute(
            "SELECT e.id, s.provider_subscription_id FROM billing_events e "
            "JOIN subscriptions s ON s.id = e.subscription_id "
            "WHERE e.new_status = 'cancelled' AND e.processed = 0"
        ).fetchall()
        for event_id, provider_subscription_id in rows:
            billing.cancel_plan(provider_subscription_id)
            conn.execute("UPDATE billing_events SET processed = 1 WHERE id = ?", (event_id,))
            conn.commit()
        time.sleep(60)
PY
cat > migrations/0007_status_audit.sql <<'SQL'
CREATE TABLE billing_events (
    id INTEGER PRIMARY KEY,
    subscription_id INTEGER NOT NULL,
    old_status TEXT,
    new_status TEXT,
    processed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TRIGGER subscription_status_audit
AFTER UPDATE OF status ON subscriptions
BEGIN
    INSERT INTO billing_events (subscription_id, old_status, new_status)
    VALUES (NEW.id, OLD.status, NEW.status);
END;
SQL
