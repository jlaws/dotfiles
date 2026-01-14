---
name: security-checklist
description: Security best practices checklist. Use when writing code that handles user input, secrets, or external data.
---

# Security Checklist

## Secrets Management

### Rules
- Never commit secrets, API keys, or credentials
- Use environment variables or secret managers
- Add sensitive patterns to `.gitignore`

### .gitignore Patterns
```gitignore
.env
.env.*
*.pem
*.key
credentials.json
secrets.yaml
```

### Environment Variables
```typescript
// Good
const apiKey = process.env.API_KEY;
if (!apiKey) throw new Error('API_KEY required');

// Bad
const apiKey = 'sk-1234567890abcdef';
```

## Input Handling

### Validate All User Input
Never trust user input. Validate and sanitize at system boundaries.

```typescript
import { z } from 'zod';

const UserInput = z.object({
  email: z.string().email(),
  age: z.number().int().positive().max(150),
  name: z.string().min(1).max(100),
});

function createUser(input: unknown) {
  const validated = UserInput.parse(input); // throws on invalid
  return db.users.create(validated);
}
```

### Parameterized Queries Only
Never concatenate SQL strings.

```typescript
// DANGEROUS: SQL injection
const query = `SELECT * FROM users WHERE id = '${userId}'`;

// Safe: parameterized
const query = 'SELECT * FROM users WHERE id = $1';
await db.query(query, [userId]);
```

### Escape Output Based on Context
- HTML context: escape `<`, `>`, `&`, `"`, `'`
- URL context: use `encodeURIComponent`
- JSON context: use `JSON.stringify`

## Dependencies

### Before Adding
```bash
npm audit          # Node.js
cargo audit        # Rust
pip-audit          # Python
```

### Keep Updated
- Enable Dependabot or similar
- Review and apply security patches promptly
- Prefer well-maintained libraries with active security teams

## Red Flags

| Pattern | Risk |
|---------|------|
| `eval()`, `exec()` | Code injection |
| `new Function()` | Code injection |
| SQL string concatenation | SQL injection |
| `dangerouslySetInnerHTML` | XSS |
| Disabled CORS | CSRF |
| Hardcoded credentials | Credential leak |
| `chmod 777` | Privilege escalation |
| Disabled SSL verification | MITM attacks |

### Code Review Checklist
- [ ] No secrets in code
- [ ] User input validated
- [ ] SQL queries parameterized
- [ ] Output properly escaped
- [ ] Dependencies audited
- [ ] Error messages don't leak internals
