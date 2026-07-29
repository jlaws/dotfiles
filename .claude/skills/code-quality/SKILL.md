---
name: code-quality
description: "Use when assessing code smells, health, or style."
allowed-tools: Read, Grep, Glob
---

# Code Quality

## Principles

| Principle | Rule |
|-----------|------|
| Composition > Inheritance | Prefer protocols/interfaces |
| Explicit > Implicit | Clarity beats cleverness |
| Favor Uniformity | One way to do each thing (test framework, build tool, deploy method). Migrate quickly + add automatic checks to prevent reversion. Easier to (re-)learn, maintain, and hand off |
| Follow Ecosystem Patterns | Go all-in on chosen framework's philosophy and idioms. Codify deviations into team policy / coding agent prompts. Fewer surprises for newcomers |
| External Configuration | Enable external config for components; follow ecosystem patterns (pydantic-settings, Spring `@ConfigurationProperties`, env vars). Easily reconfigured across environments and tests |

## Working vs Production-Ready

| Working | Production-Ready |
|---|---|
| Happy path works | Error paths handled |
| Manual testing | Automated tests |
| Hardcoded config | Externalized config |

- Distinguish explicitly when delivering: "This works but needs X before production"
- Propose scope cuts explicitly — never implement them silently

## Code Smells Checklist

**Functions**
- Single responsibility, <30 lines
- Max 3 parameters; use parameter object beyond that

**Complexity**
- Max 2 levels nesting; use early returns

**Make Invalid States Unrepresentable**
- Use generics / type hints to catch issues at compile-time / static analysis (Python `list[str]`, Java `List<String>`, TS `Array<string>`)
- Use specialized types where invalid inputs are unrepresentable (pydantic `BaseModel` with `dict[str, str]` over raw `str`, typed keys over string constants)
- No `any` in TypeScript (use `unknown`); no force unwraps in Swift (unless provably safe)
- Use `Optional` / `Option` for null safety -- never return bare `None`/`null` when absence is possible
- Validate early at boundaries, convert to constrained types, pass constrained types downstream
- Leverage utility types: `Pick`, `Omit`, `Partial`, `NonNullable`
- Priority: **compile-time > static analysis > runtime** for catching errors

## Anti-Patterns

**Code**
- **Premature abstraction** -- wait for 2+ concrete implementations

**Process**
- **TODOs**: acceptable only with a ticket and a reason, e.g. `// TODO(#123): handle rate limiting`. Prefer filing a ticket over leaving a bare marker. A bare `// TODO`, or one standing in for work that was in scope, is not acceptable.

## Style Defaults

| Rule | Value |
|------|-------|
| Line endings | LF (Unix) |
| Test location | Colocated (`foo.ts` + `foo.test.ts`) or parallel (`src/` + `tests/`) |

**Naming conventions:** JS/TS/Swift = `camelCase`, Python/Rust/Go = `snake_case`, Types = `PascalCase`, Constants = `SCREAMING_SNAKE_CASE`

### Style Guides by Language

| Language | Style Guide |
|----------|-------------|
| Python | Google Python Style Guide |
| JavaScript/TypeScript | Google JS + TS Style Guides |
| Go | Google Go Style Guide |
| Bash | Google Shell Style Guide |
| Rust | Rust Style Guide |
| C#/.NET | Microsoft C# Coding Conventions |

See each language skill for detailed naming and practice rules.

**Import order** (separated by blank lines): 1. Standard library, 2. Third-party, 3. Local modules

## Lint Priority Triage

| Priority | Examples | When to Fix |
|----------|----------|-------------|
| High | Type errors blocking build, security vulns, runtime errors | Immediately |
| Medium | Missing type annotations, unused vars, style violations | Before commit |
| Low | Formatting inconsistencies, comment improvements | When convenient |

**Safe auto-fixes:** `prettier --write .`, `eslint --fix .`
**Manual fixes needed:** type annotations, logic errors, missing error handling, accessibility

## Refactoring Decision Framework

- **Early returns** over nested conditionals
- **Parameter objects** when >3 params
- **Lookup maps** over conditional chains
- **Extract function** when a block needs a comment to explain intent
- **Typed errors** over generic catch-all

```typescript
// Early return pattern — flatten nested conditionals
function processOrder(order: Order): Result {
  if (!order.items.length) return Result.empty();
  if (!order.payment)      return Result.error('No payment');
  if (order.total <= 0)    return Result.error('Invalid total');

  return Result.ok(checkout(order));
}
```

## Performance (Profile First)

**React/Next.js**: `React.memo`, `useMemo`, code splitting, virtual scrolling
**Database**: Index frequently queried fields, batch queries (N+1), pagination
**API**: SWR/React Query caching, debounce/throttle, parallel requests
**Bundle**: Tree-shake, dynamic imports, route-level code splitting

## Dead Code Removal

- Unused imports, unreachable code, unused variables
- Run `tsc --noEmit` and check lint warnings

## Measurement Tools

| Layer | Tools |
|-------|-------|
| Frontend | Chrome DevTools, Lighthouse CI, React Profiler, Bundle Analyzer |
| Backend | Node.js profiler, DB query analyzer, APM (DataDog/New Relic), k6/Artillery |
