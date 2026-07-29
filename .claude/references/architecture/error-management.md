# Error Management

What this file owns: error tracking setup, correlation IDs, log structure, reading stack traces, and
grouping errors into distinct issues.

What it does not own, because a sibling says it better and used to disagree with this file:

| Topic | Owner |
|---|---|
| Severity levels and response times | `devops/incident-management.md` (SEV1-SEV4, with response times) |
| Incident response phases, postmortems, Five Whys | `devops/incident-management.md` |
| Alert routing, thresholds, burn rates | `devops/observability.md` |
| Retry and backoff | `architecture/retry-patterns.md` (which correctly refuses to retry 400/401/403/404/422 and honors `Retry-After`) |
| Root-cause investigation method | the `debugging-methodology` skill |
| Fail-fast and input validation | `architecture/error-handling-patterns.md` |

## Error Tracking Service Integration

### Sentry (Node.js/Express)

Two parts are worth getting right; the rest of the setup is whatever the SDK docs say.

```javascript
Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV,
  tracesSampleRate: 0.1,

  // Strip credentials before they leave the process. Sentry keeps whatever you send.
  beforeSend(event) {
    delete event.request?.cookies;
    if (event.request?.headers) delete event.request.headers.authorization;
    return event;
  },

  // Group by the thing that broke, not by the message, which usually carries an id.
  beforeSendTransaction(event) {
    event.fingerprint = ['{{ default }}', event.transaction ?? 'unknown'];
    return event;
  },
});
```

## Structured Logging

### Log Schema

One shape for every log line, so queries work across services:

| Field | Purpose |
|---|---|
| `timestamp` | ISO 8601, UTC |
| `level` | `error`, `warn`, `info`, `debug` |
| `service`, `version` | Which build produced this |
| `trace_id`, `correlation_id` | Joins a line to a request and a distributed trace |
| `user_id` | Whose request, when known |
| `error.type`, `error.message`, `error.stack` | Nested so the fields stay queryable |
| `context` | Everything task-specific, namespaced under one key |

Inject `trace_id` into every line rather than logging it separately — see
`devops/observability.md` for the tracing side.

## Correlation ID Pattern

Accept an inbound id or mint one, echo it on the response, and forward it on every outbound call.
Without the forward, traces break at the first service boundary.

```javascript
const { v4: uuidv4 } = require('uuid');

function correlationIdMiddleware(req, res, next) {
    req.correlationId = req.headers['x-correlation-id'] || uuidv4();
    res.setHeader('x-correlation-id', req.correlationId);
    next();
}

// The id has to be passed in; it is per-request state, not module state.
async function makeApiCall(url, data, correlationId) {
    return axios.post(url, data, {
        headers: { 'x-correlation-id': correlationId }
    });
}
```

## Error Classification

### By Type
- **Runtime**: Exceptions, crashes, null dereferences
- **Logic**: Wrong behavior, wrong calculations
- **Integration**: A dependency failed or changed shape
- **Resource**: Out of memory, disk, connections, file handles
- **Configuration**: Missing or wrong environment values

### By Reproducibility
- **Deterministic**: Same input, same failure. Cheapest to fix.
- **Intermittent**: Timing, ordering, or load dependent. Needs the conditions captured before the fix.
- **Environment-specific**: Works here, fails there. The difference is the bug.

Reproducibility drives the approach more than severity does: a deterministic critical bug is often
faster to fix than an intermittent minor one.

## Stack Trace Analysis Patterns

The useful move is finding the first frame you own; the top frame is usually the framework reporting
someone else's mistake.

```
# Null dereference deep in a framework
NullPointerException at java.util.HashMap.hash
--> Your code passed null in. Look at your own nearest frame, not HashMap.

# Timeout after a long wait
TimeoutException after 30000ms at okhttp3.Http2Stream.waitForIo
--> A dependency is slow, not broken. Needs a timeout budget and a circuit breaker.

# Race condition
ConcurrentModificationException at ArrayList$Itr.checkForComodification
--> A collection was mutated while iterating it. The trace shows the reader, not the writer.
```

## Error Grouping / Fingerprinting

Distinct errors get separate issues; the same error with different ids does not. Normalize the
variable parts out of the message before hashing it:

```python
import hashlib
import re

def normalize_message(message: str) -> str:
    """Strip the parts that differ per occurrence so grouping survives them."""
    message = re.sub(r"0x[0-9a-fA-F]+", "<addr>", message)
    message = re.sub(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        "<uuid>",
        message,
    )
    message = re.sub(r"https?://\S+", "<url>", message)
    return re.sub(r"\b\d+\b", "<number>", message)


def fingerprint(error_type: str, message: str, top_frame: str) -> str:
    key = f"{error_type}|{normalize_message(message)}|{top_frame}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]
```

Order matters: normalize UUIDs and addresses before the bare-number pass, or the digits inside them
get replaced first and the groups split anyway.

Sentry does this natively — hand-roll it only when aggregating logs yourself.
