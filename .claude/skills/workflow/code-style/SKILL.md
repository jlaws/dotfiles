---
name: code-style
description: Universal code formatting and naming conventions. Use when writing or reviewing code.
---

# Code Style Guide

## Formatting

| Rule | Value |
|------|-------|
| Indentation | 2 spaces (no tabs) |
| Line endings | LF (Unix) |
| Charset | UTF-8 (no BOM) |
| Trailing whitespace | Trim |
| Final newline | Always |
| Line length | 80-100 soft limit |

## Naming Conventions

### Variables & Functions
- **JS/TS/Swift**: `camelCase`
- **Python/Rust/Go**: `snake_case`

### Types & Classes
- All languages: `PascalCase`

### Constants
- `SCREAMING_SNAKE_CASE` or language idiom

### Booleans
Prefix with: `is`, `has`, `can`, `should`

```typescript
const isEnabled = true;
const hasPermission = user.roles.includes('admin');
const canEdit = isEnabled && hasPermission;
```

### Functions
Prefix with verb: `get`, `set`, `create`, `handle`, `fetch`, `update`, `delete`

```typescript
function getUserById(id: string): User { }
function createOrder(items: Item[]): Order { }
function handleSubmit(event: FormEvent): void { }
```

## Organization

### Import Grouping
Separate with blank lines:
1. Standard library
2. Third-party packages
3. Local modules

```typescript
// stdlib
import { readFile } from 'fs/promises';

// third-party
import express from 'express';
import { z } from 'zod';

// local
import { config } from './config';
import { UserService } from './services/user';
```

```python
# stdlib
import os
from pathlib import Path

# third-party
import requests
from pydantic import BaseModel

# local
from .config import settings
from .models import User
```

### File Size
- Keep files under 300 lines
- One concept per file when practical

### Test Location
- Colocated with source (`foo.ts` + `foo.test.ts`)
- Or parallel directory (`src/` + `tests/`)
