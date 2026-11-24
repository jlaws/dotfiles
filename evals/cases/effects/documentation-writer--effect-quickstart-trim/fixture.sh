set -euo pipefail
mkdir -p widgetkit docs
cat > widgetkit/__init__.py <<'PY'
"""widgetkit: a client for the Widgets API."""
from widgetkit.client import Client, Widget

__all__ = ["Client", "Widget"]
PY
cat > widgetkit/client.py <<'PY'
"""Public client for the Widgets API."""
import logging
from dataclasses import dataclass


@dataclass
class Widget:
    id: str
    name: str
    type: str


class Client:
    def __init__(self, api_key, timeout=30.0, max_retries=2):
        """Create a client. api_key is required; timeout is in seconds."""
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._cache = None
        self._log = logging.getLogger("widgetkit")
        self._next = 1

    def create_widget(self, name, type="counter"):
        """Create a widget and return it. type is counter, gauge, or chart."""
        if type not in ("counter", "gauge", "chart"):
            raise ValueError(f"unknown widget type: {type}")
        widget = Widget(id=f"wgt_{self._next:03d}", name=name, type=type)
        self._next += 1
        self._log.debug("created %s", widget.id)
        return widget

    def enable_cache(self, redis_url):
        """Cache widget reads in Redis."""
        self._cache = redis_url

    def register_webhook(self, url, events):
        """Send the named widget events to url."""
        return {"url": url, "events": list(events)}
PY
cat > docs/configuration.md <<'MD'
# Configuration

Covers logging, timeouts and retries (`Client(timeout=..., max_retries=...)`), and Redis caching
(`client.enable_cache(redis_url)`).
MD
cat > docs/webhooks.md <<'MD'
# Webhooks

Covers `client.register_webhook(url, events)` and the widget events you can subscribe to.
MD
cat > docs/quickstart.md <<'MD'
# Quickstart

## Step 1: Check your Python version

You need Python 3.9 or later.

## Step 2: Install

```bash
pip install widgetkit
```

## Step 3: Get an API key

Create a key on the dashboard under Settings, then API keys.

## Step 4: Set the API key

```bash
export WIDGETKIT_API_KEY=your_key_here
```

## Step 5: Turn on debug logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Step 6: Tune timeouts and retries

```python
client = Client(api_key=key, timeout=10, max_retries=5)
```

## Step 7: Create a widget

```python
from widgetkit import Client

client = Client(api_key=os.environ["WIDGETKIT_API_KEY"])
widget = client.create_widget(name="Daily Signups", type="counter")
# ... handle errors here
print(widget.id)
```

## Step 8: Enable caching

```python
client.enable_cache("redis://localhost:6379/0")
```

## Step 9: Register a webhook

```python
client.register_webhook("https://example.com/hooks", events=[...])
```
MD
