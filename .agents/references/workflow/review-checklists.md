# Review Checklists (Supplementary)

Structured checklists for code review, supplementing the main `skills/code-review-patterns` skill. Use during Step 4 (Multi-Perspective Analysis) for domain-specific coverage.

## SQL Safety

| Pattern | Detection Signal | Fix |
|---------|-----------------|-----|
| Non-parameterized queries | String interpolation/concatenation in SQL | Use parameterized queries / prepared statements |
| No-lock DDL | `ALTER TABLE` without `CONCURRENTLY` (Postgres) or equivalent | Use online DDL; schedule during low traffic |
| N+1 queries | DB call inside a loop over results | Batch query / eager load / JOIN |
| Missing transaction boundaries | Multi-step mutations without BEGIN/COMMIT | Wrap in transaction; consider idempotency |
| Missing index | WHERE/JOIN on unindexed column with large table | Add index; verify with EXPLAIN |

## Race Conditions

| Pattern | Detection Signal | Fix |
|---------|-----------------|-----|
| Shared mutable state | Global/static variable written from multiple threads/goroutines | Mutex, channel, or atomic operation |
| Lock ordering inconsistency | Multiple locks acquired in different orders across call sites | Establish canonical lock order; document it |
| TOCTOU | Check (e.g., file exists) then act (open file) without atomicity | Use atomic operations (O_CREAT\|O_EXCL, compare-and-swap) |
| Async hazards | Missing `await`, unhandled promise rejection, goroutine leak | Lint rules; context cancellation; `Promise.all` error handling |
| Check-then-act | `if !map.has(key) { map.set(key, val) }` without lock | Use `putIfAbsent` / `setdefault` / sync primitive |

## LLM Trust Boundaries

| Pattern | Detection Signal | Fix |
|---------|-----------------|-----|
| LLM output in SQL/shell/eval | String from LLM response passed to `exec`/`eval`/raw SQL | Treat as untrusted input; parameterize or sandbox |
| No output validation | LLM text rendered as HTML without sanitization | Sanitize/escape before rendering |
| Missing rate/cost limits | LLM API calls without budget cap or rate limiter | Add per-user/per-request cost ceiling and rate limit |
| Prompt injection | User input concatenated directly into prompt template | Separate system/user messages; input validation |
| Trusting LLM-generated URLs/paths | File path or URL from LLM used in fetch/open | Allowlist validation; never trust blindly |

## Enum / Exhaustive Matching

| Pattern | Detection Signal | Fix |
|---------|-----------------|-----|
| Non-exhaustive switch | `switch` without `default` or missing enum case | Enable exhaustiveness lint; add `default: assertNever(x)` |
| New variant not propagated | Enum extended but consumers not updated | Grep for all switch/match on that type; update each |
| String-based dispatch | `if (type === "foo")` instead of typed enum | Refactor to enum/union type |
| Unknown value in deserialization | API sends new enum value; client crashes | Handle unknown gracefully; log + fallback |

## Design System Compliance

| Pattern | Detection Signal | Fix |
|---------|-----------------|-----|
| Hardcoded colors/spacing | Hex codes or pixel values instead of tokens | Use design tokens / theme variables |
| Duplicate component | Custom component reimplements existing DS component | Use existing component; extend if needed |
| Inconsistent component API | Different prop naming across similar components | Align with DS naming conventions |
| Missing dark mode | No theme-aware styling; hardcoded light colors | Use semantic color tokens |
| Accessibility gap | Missing aria-label, insufficient contrast ratio | Add labels; check WCAG AA (4.5:1 text, 3:1 large) |

## Documentation Currency

| Pattern | Detection Signal | Fix |
|---------|-----------------|-----|
| Stale API/CLI docs | Public signature/flag changed in diff, docs untouched | Update reference docs; add CHANGELOG entry |
| Renamed public symbol | Rename in diff, old name still in README/docs | Grep docs for old name; update |
| New feature undocumented | New user-facing capability, no README/usage update | Add usage section |
| KB asset added without registration | New skill/command/agent, CLAUDE.md structure / MEMORY index untouched, mirror unsynced | Register + sync `.agents` ↔ `.claude` |
