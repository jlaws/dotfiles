---
name: nodejs-backend-patterns
description: Node.js backend architecture opinions and non-obvious patterns. Use when building Node.js APIs or microservices.
---

# Node.js Backend Patterns

Architecture decisions and gotchas for Node.js backends.

## Style Guide

Source: Google JS + TS Style Guides. Extends JS/TS Patterns skill conventions.

- **Naming**: follow JS/TS Patterns skill; abbreviations as words (`httpServer` not `HTTPServer`)
- Named exports only (no default exports)
- Throw only `Error` objects, catch as `unknown`

## Framework Decision

| Framework | When |
|-----------|------|
| **Express** | Team knows it, large middleware ecosystem needed |
| **Fastify** | New projects, need performance, want built-in validation |
| **Hono** | Edge/serverless, multi-runtime (Bun, Deno, CF Workers) |

## Architecture: Layered (always)

```
controllers/  -- parse request, send response, delegate to service
services/     -- business logic, orchestrate repositories
repositories/ -- data access only
middleware/   -- auth, validation, logging, error handling
```

Never put business logic in controllers or routes.

## Error Handling Pattern

```typescript
// Custom errors with status codes
class AppError extends Error {
  constructor(public message: string, public statusCode = 500, public isOperational = true) {
    super(message);
  }
}
class NotFoundError extends AppError { constructor(msg = 'Not found') { super(msg, 404); } }
class ValidationError extends AppError { constructor(msg: string, public errors?: any[]) { super(msg, 400); } }

// Global error handler (Express)
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  if (err instanceof AppError) {
    return res.status(err.statusCode).json({ error: err.message });
  }
  logger.error({ error: err.message, stack: err.stack });
  res.status(500).json({ error: process.env.NODE_ENV === 'production' ? 'Internal error' : err.message });
});

// Async wrapper (no try/catch in every handler)
const asyncHandler = (fn: Function) => (req: Request, res: Response, next: NextFunction) =>
  Promise.resolve(fn(req, res, next)).catch(next);
```

## Validation: Zod (default choice)

```typescript
import { z } from 'zod';

const createUserSchema = z.object({
  body: z.object({
    name: z.string().min(1),
    email: z.string().email(),
    password: z.string().min(8),
  }),
});

// Middleware
const validate = (schema: AnyZodObject) => async (req: Request, res: Response, next: NextFunction) => {
  try { await schema.parseAsync({ body: req.body, query: req.query, params: req.params }); next(); }
  catch (e) { next(new ValidationError('Validation failed', (e as ZodError).errors)); }
};
```

## Database Gotchas

- **Always use connection pooling** -- `max: 20`, `idleTimeoutMillis: 30000`
- **Transactions**: `BEGIN` -> operations -> `COMMIT`/`ROLLBACK`, always in `try/finally` with `client.release()`
- **Parameterized queries always** -- `$1, $2` (Postgres) or `?` (MySQL), never string interpolation

## Security Checklist

- `helmet()` for HTTP headers
- CORS: never `origin: '*'` in production
- Rate limiting: `express-rate-limit` with Redis store
- Input validation at API boundary (Zod)
- JWT: short-lived access tokens (15min) + refresh tokens (7d)

## Key Opinions

- **TypeScript always** -- no plain JS for backends
- **Structured logging** -- `pino` (faster than Winston)
- **`node:` prefix** for built-in modules -- `import { readFile } from 'node:fs/promises'`
- **Graceful shutdown** -- handle SIGTERM, drain connections
- **Health check endpoint** -- `/health` returning `{ status: 'ok' }`
