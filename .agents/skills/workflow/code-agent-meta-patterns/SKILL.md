---
name: code-agent-meta-patterns
description: "Code agent workflow optimization including CLAUDE.md design, hooks configuration, and context management. Use when optimizing code agent workflows, designing CLAUDE.md files, or configuring hooks. Do NOT use for skill creation or editing workflow (use writing-skills)."
---

# Code Agent Meta-Patterns

## CLAUDE.md Optimization

### What to Include vs. Avoid

| Include | Avoid |
|---------|-------|
| Project-specific conventions (naming, patterns) | Generic programming advice the agent already knows |
| Build/test/lint commands | Full API documentation (link to it instead) |
| Architecture decisions and rationale | Verbose explanations of obvious patterns |
| File/directory purpose map | Repeating info available in package.json/pyproject.toml |
| Aliases and shortcuts developers use | Instructions that change weekly |
| Non-obvious constraints ("never modify X") | Copy-pasted README content |
| Communication style preferences | Long lists of technologies used |

### Layering Strategy

CLAUDE.md files cascade. Higher specificity wins.

```
~/.claude/CLAUDE.md              # Global: communication style, git conventions
project/CLAUDE.md                # Project: build commands, architecture, file map
project/.claude/CLAUDE.md        # Project (alt location): same as above
project/src/api/CLAUDE.md        # Directory: API-specific patterns, conventions
```

**Rules:**
- Global: personal preferences, style, universal workflow rules
- Project root: build/test/lint commands, project structure, tech stack decisions
- Subdirectory: module-specific conventions, local patterns, "how this subsystem works"
- Keep each file under 200 lines; link to detailed docs instead of inlining

### Effective CLAUDE.md Template

```markdown
# CLAUDE.md

## Build & Test
\`\`\`bash
npm run build          # TypeScript → dist/
npm test               # Jest, ~30s
npm run lint           # ESLint + Prettier
npm run test:e2e       # Playwright, requires running server
\`\`\`

## Architecture
- Monorepo: apps/ (Next.js) + packages/ (shared libs)
- API: tRPC routers in apps/api/src/routers/
- DB: Drizzle ORM, migrations in packages/db/migrations/

## Conventions
- Barrel exports from every package (index.ts)
- Zod schemas co-located with routers
- Error handling: use AppError class, never throw raw strings
- Feature flags: check packages/flags/ before adding conditionals

## Key Files
| File | Purpose |
|------|---------|
| `turbo.json` | Build pipeline config |
| `packages/db/schema.ts` | Database schema source of truth |
| `.env.example` | Required env vars with descriptions |
```

## Skill Design

### Frontmatter Conventions

```yaml
---
name: kebab-case-name           # Matches directory name
description: "Use when [specific trigger scenario]. Also applies to [related scenarios]."
---
```

**Description rules:**
- Always start with "Use when" -- this is the trigger phrase the agent matches on
- Be specific: "Use when writing database migrations" not "Use for database stuff"
- Include 2-3 trigger scenarios separated by commas or "or"
- Keep under 200 characters

### Skill Structure Pattern

Follow this order for consistent, scannable skills:

```markdown
# Skill Title

## Decision Table
(When to use which approach — always first)

## Core Patterns
(The main content, with code examples)

## Code Examples
(Copy-pasteable, realistic examples)

## Gotchas
(Non-obvious failure modes — always last)
```

### Sizing Guidelines

| Skill Scope | Target Lines | Example |
|---|---|---|
| Narrow (one task) | 80-150 | "Writing database migrations" |
| Medium (workflow) | 150-250 | "API design patterns" |
| Broad (discipline) | 250-350 | "System design interviews" |

Over 350 lines: split into multiple skills.

### Description Trigger Examples

| Good | Bad |
|------|-----|
| "Use when designing REST APIs, choosing HTTP methods, or structuring URL hierarchies" | "REST API skill" |
| "Use when writing unit tests for React components using Testing Library" | "Use for testing" |
| "Use when debugging production incidents, writing postmortems, or setting up alerting" | "Incident management" |

## Context Management

### Keeping Context Small

| Strategy | When | How |
|----------|------|-----|
| Link, don't inline | Docs > 50 lines | "See docs/architecture.md for details" |
| Scope CLAUDE.md | Always | Only include what affects daily coding |
| Prune stale instructions | Monthly | Remove anything that hasn't been relevant |
| Use directory-level CLAUDE.md | Large monorepos | Put module-specific rules in module dirs |
| Search then read | Always | Search first, read only confirmed-relevant files |
| Clean external content | WebFetch, logs | Strip HTML boilerplate, nav, ads before reasoning |

### Context Budget Rules
- CLAUDE.md files: aim for <150 lines each
- Skills: 150-300 lines typical
- If a skill needs >350 lines, it's trying to do too much
- Prefer tables and code over prose (higher information density)
## Gotchas

- **Context bloat**: Every line in CLAUDE.md consumes context window. Ruthlessly prune. If the agent already knows it (general programming, language syntax), don't restate it.
- **Over-engineering skills**: A skill should solve a recurring problem. If you've only needed it once, it's not a skill -- it's a conversation. Wait for the third occurrence.
- **Stale instructions**: CLAUDE.md that references deleted files, old conventions, or deprecated workflows actively misleads. Review quarterly.
- **Description mismatch**: If the skill description doesn't match what the skill actually does, the agent will invoke it at the wrong time or skip it when needed. Test trigger phrases.
- **Skill overlap**: Two skills with similar descriptions cause unpredictable invocation. Deduplicate or make descriptions clearly distinct.
- **Ignoring layer precedence**: Putting project-specific rules in global CLAUDE.md means they apply to every project. Keep global truly global.
