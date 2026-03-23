# Error Management

Comprehensive error tracking, analysis, alerting, incident response, and prevention patterns.

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

async function makeApiCall(url, data) {
    return axios.post(url, data, {
        headers: { 'x-correlation-id': req.correlationId }
    });
}
```

## Error Classification

### By Severity
- **Critical**: System down, data loss, security breach
- **High**: Major feature broken, significant user impact
- **Medium**: Partial degradation, workarounds available
- **Low**: Minor bugs, cosmetic issues

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

## Root Cause Analysis

### The Five Whys

```
Error: Database connection timeout after 30s
Why? Connection pool was exhausted
Why? All connections held by long-running queries
Why? New feature introduced N+1 query patterns
Why? ORM lazy-loading not properly configured
Why? Code review didn't catch the performance regression
```

### Systematic Investigation
1. Reproduce the error with minimal steps
2. Isolate the failure point (exact line/component)
3. Analyze the call chain leading to failure
4. Inspect variable state at failure point
5. Review git history for recent changes
6. Test hypotheses with targeted experiments

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

## Alert Configuration

```python
alert_rules = [
    {
        'name': 'High Error Rate',
        'condition': 'error_rate > 0.05',
        'window': '5m',
        'severity': 'critical',
        'channels': ['slack', 'pagerduty']
    },
    {
        'name': 'Response Time Degradation',
        'condition': 'response_time_p95 > 1000',
        'window': '10m',
        'severity': 'warning',
        'channels': ['slack']
    },
    {
        'name': 'Memory Usage Critical',
        'condition': 'memory_percent > 90',
        'window': '5m',
        'severity': 'critical',
        'channels': ['slack', 'pagerduty']
    }
]
```

### Alert Manager with Cooldown

```python
class AlertManager:
    async def evaluate_rules(self, metrics):
        for rule in self.rules:
            if await self._should_alert(rule, metrics):
                await self._send_alert(rule, metrics)

    async def _should_alert(self, rule, metrics):
        if not self._check_threshold(metrics[rule.condition], rule.threshold):
            return False
        last_alert = self.alert_history.get(rule.name)
        if last_alert and datetime.now() - last_alert < rule.cooldown:
            return False
        return True
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

## Retry with Exponential Backoff

```typescript
async function retryWithBackoff<T>(
    fn: () => Promise<T>,
    options = { maxAttempts: 3, baseDelayMs: 1000, maxDelayMs: 30000 }
): Promise<T> {
    let lastError: Error;
    for (let attempt = 0; attempt < options.maxAttempts; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            if (attempt < options.maxAttempts - 1) {
                const delay = Math.min(
                    options.baseDelayMs * Math.pow(2, attempt),
                    options.maxDelayMs
                );
                const jitter = Math.random() * 0.1 * delay;
                await new Promise(resolve => setTimeout(resolve, delay + jitter));
            }
        }
    }
    throw lastError;
}
```

## Incident Response Workflow

### Phase 1: Detection and Triage (0-5 min)
1. Acknowledge alert/incident
2. Assess severity and user impact
3. Assign incident commander
4. Create incident channel
5. Update status page if customer-facing

### Phase 2: Investigation (5-30 min)
1. Gather observability data (errors, traces, logs, metrics)
2. Correlate with recent changes (deployments, configs)
3. Form initial hypothesis
4. Document findings

### Phase 3: Mitigation (Immediate)
1. Implement fix: rollback, scale up, disable feature flag, or hotfix
2. Verify mitigation worked
3. Monitor for stability

### Phase 4: Post-Incident
1. Schedule postmortem (within 48 hours)
2. Create detailed timeline
3. Identify true root cause
4. Create prevention action items

## Prevention: Input Validation

```typescript
import { z } from 'zod';

const PaymentSchema = z.object({
    amount: z.number().positive().max(1000000),
    currency: z.enum(['USD', 'EUR', 'GBP']),
    customerId: z.string().uuid(),
    paymentMethodId: z.string().min(1)
});

function processPayment(request: unknown) {
    const validated = PaymentSchema.parse(request);
    return chargeCustomer(validated);
}
```

## Prevention: Error Boundaries (React)

```typescript
class ErrorBoundary extends Component<Props, State> {
    static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        Sentry.captureException(error, {
            contexts: { react: { componentStack: errorInfo.componentStack } }
        });
    }

    render() {
        if (this.state.hasError) {
            return this.props.fallback || <ErrorFallback error={this.state.error} />;
        }
        return this.props.children;
    }
}
```

## Prevention: Static Analysis Rules

```yaml
# ESLint rules
rules:
  no-floating-promises: error
  no-unhandled-error: error
  require-await: error
  no-unused-catch-bindings: error

# TypeScript strict mode
compilerOptions:
  strict: true
  noImplicitAny: true
  strictNullChecks: true
```
