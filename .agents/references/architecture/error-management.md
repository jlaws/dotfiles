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

```javascript
import * as Sentry from "@sentry/node";

Sentry.init({
    dsn: process.env.SENTRY_DSN,
    environment: process.env.NODE_ENV,
    release: process.env.GIT_COMMIT_SHA,
    tracesSampleRate: 0.1,

    beforeSend: (event, hint) => {
        if (event.request?.cookies) delete event.request.cookies;

        if (hint.originalException) {
            event.fingerprint = [
                hint.originalException.name,
                extractLocation(hint.originalException.stack)
            ];
        }
        return event;
    },

    integrations: [
        new Sentry.Integrations.Http({ tracing: true }),
        new Sentry.Integrations.Express({ app })
    ]
});

process.on('uncaughtException', (error) => {
    Sentry.captureException(error, { level: 'fatal' });
    gracefulShutdown();
});

process.on('unhandledRejection', (reason) => {
    Sentry.captureException(reason, { tags: { type: 'unhandled_rejection' } });
});
```

## Structured Logging

```typescript
import winston from 'winston';

class StructuredLogger {
    private logger: winston.Logger;

    constructor(config: LoggerConfig) {
        this.logger = winston.createLogger({
            level: config.level || 'info',
            format: winston.format.combine(
                winston.format.timestamp(),
                winston.format.errors({ stack: true }),
                winston.format.json()
            ),
            defaultMeta: {
                service: config.service,
                environment: config.environment,
                version: config.version
            },
            transports: [
                new winston.transports.Console(),
                new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
                new winston.transports.File({ filename: 'logs/combined.log' })
            ]
        });
    }

    error(message: string, error?: Error, context?: any) {
        this.logger.error(message, {
            error: { message: error?.message, stack: error?.stack, name: error?.name },
            ...context
        });
    }
}
```

### Log Schema

```json
{
  "timestamp": "2025-01-03T14:23:45.123Z",
  "level": "ERROR",
  "correlation_id": "req-7f3b2a1c-4d5e-6f7g",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "service": "payment-service",
  "environment": "production",
  "error": {
    "type": "PaymentProcessingException",
    "message": "Failed to charge card",
    "stack_trace": "...",
    "fingerprint": "payment-card-failure"
  },
  "request": {
    "method": "POST",
    "path": "/api/payments/charge",
    "duration_ms": 2547
  }
}
```

## Correlation ID Pattern

```javascript
const { v4: uuidv4 } = require('uuid');

function correlationIdMiddleware(req, res, next) {
    const correlationId = req.headers['x-correlation-id'] || uuidv4();
    req.correlationId = correlationId;
    res.setHeader('x-correlation-id', correlationId);
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
- **Runtime**: Exceptions, crashes, null pointer errors
- **Logic**: Incorrect behavior, wrong calculations
- **Integration**: API failures, network timeouts
- **Performance**: Memory leaks, slow queries
- **Configuration**: Missing env vars, invalid settings
- **Security**: Auth failures, injection attempts

### By Reproducibility
- **Deterministic**: Consistently reproducible
- **Intermittent**: Race conditions, timing issues
- **Environmental**: Specific to certain configs
- **Load-dependent**: Under high traffic only

## Stack Trace Analysis Patterns

```
# Null Pointer Deep in Framework
NullPointerException at java.util.HashMap.hash
--> Application passed null to framework. Focus on your code frame.

# Timeout After Long Wait
TimeoutException after 30000ms at okhttp3.Http2Stream.waitForIo
--> External service slow. Need retry logic and circuit breaker.

# Race Condition
ConcurrentModificationException at ArrayList$Itr.checkForComodification
--> Collection modified while iterating. Need thread-safe structures.
```

## Error Grouping / Fingerprinting

```python
class ErrorGrouper:
    def generate_fingerprint(self, error):
        normalized_message = self.normalize_message(error['message'])
        components = [
            error.get('type', 'Unknown'),
            normalized_message,
            self.extract_location(error.get('stack', ''))
        ]
        return hashlib.sha256('|'.join(components).encode()).hexdigest()[:16]

    def normalize_message(self, message):
        normalized = re.sub(r'\b\d+\b', '<number>', message)
        normalized = re.sub(r'[a-f0-9-]{36}', '<uuid>', normalized)
        normalized = re.sub(r'https?://[^\s]+', '<url>', normalized)
        return normalized.strip()
```

