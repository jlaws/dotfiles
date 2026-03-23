---
name: code-agent-meta-patterns
description: "Code agent workflow optimization including AGENTS.md design, hooks configuration, and multi-agent orchestration. Use when optimizing Codex workflows, designing AGENTS.md files, configuring hooks, or orchestrating multi-agent patterns. Do NOT use for skill creation or editing workflow (use writing-skills)."
compatibility: codex
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Codex Meta-Patterns

## AGENTS.md Optimization

### What to Include vs. Avoid

| Include | Avoid |
|---------|-------|
| Project-specific conventions (naming, patterns) | Generic programming advice Codex already knows |
| Build/test/lint commands | Full API documentation (link to it instead) |
| Architecture decisions and rationale | Verbose explanations of obvious patterns |
| File/directory purpose map | Repeating info available in package.json/pyproject.toml |
| Aliases and shortcuts developers use | Instructions that change weekly |
| Non-obvious constraints ("never modify X") | Copy-pasted README content |
| Communication style preferences | Long lists of technologies used |

### Layering Strategy

AGENTS.md files cascade. Higher specificity wins.

```
~/.codex/AGENTS.md               # Global: communication style, git conventions
project/AGENTS.md                # Project: build commands, architecture, file map
project/.agents.md               # Optional fallback if configured
project/src/api/AGENTS.md        # Directory: API-specific patterns, conventions
```

**Rules:**
- Global: personal preferences, style, universal workflow rules
- Project root: build/test/lint commands, project structure, tech stack decisions
- Subdirectory: module-specific conventions, local patterns, "how this subsystem works"
- Keep each file under 200 lines; link to detailed docs instead of inlining

### Effective AGENTS.md Template

```markdown
# AGENTS.md

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
- Always start with "Use when" -- this is the trigger phrase Codex matches on
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

## Command Design

Commands are thin wrappers that invoke skills. Keep them minimal.

### Command Structure

```markdown
---
name: review
description: "Review code changes for quality, security, and convention compliance"
---

Use the code-review skill to review the current changes.

Focus on: $ARGUMENTS

If no arguments provided, review all staged/unstaged changes.
```

### `$ARGUMENTS` Patterns

| Pattern | Use Case | Example Invocation |
|---------|----------|--------------------|
| Pass-through | Skill needs user context | `/review focus on security` |
| Optional | Skill has sensible defaults | `/deploy` or `/deploy staging` |
| Structured | Multiple params needed | `/create-pr base=main title="Fix auth"` |

### Rules
- One command = one skill invocation (usually)
- Codex does not auto-discover a custom command folder the way some other agents do; keep reusable prompts in `.codex/commands/`
- No logic in commands; all logic lives in skills
- Description should make the command discoverable

## Hook Patterns

See `references/workflow/hook-patterns.md` for full examples (PreToolUse, PostToolUse JSON configs).

**Key principles:** hooks should be fast (<5s), fail loudly, use narrow matchers, test manually first.

## Context Management

### Keeping Context Small

| Strategy | When | How |
|----------|------|-----|
| Link, don't inline | Docs > 50 lines | "See docs/architecture.md for details" |
| Scope AGENTS.md | Always | Only include what affects daily coding |
| Prune stale instructions | Monthly | Remove anything that hasn't been relevant |
| Use directory-level AGENTS.md | Large monorepos | Put module-specific rules in module dirs |
| Search then read | Always | Glob/Grep first, Read only confirmed-relevant files |
| Clean external content | WebFetch, logs | Strip HTML boilerplate, nav, ads before reasoning |
| U-shaped placement | AGENTS.md, skills | Critical content at beginning/end, reference in middle |

### When to Use Subagents (Task Tool)

| Use Subagent | Stay in Main Thread |
|---|---|
| Independent research task | Sequential dependent steps |
| Exploring a large codebase area | Editing a specific file |
| Generating boilerplate for N items | Making a single targeted change |
| Parallel investigation of alternatives | Simple question-answer |

### Context Budget Rules
- AGENTS.md files: aim for <150 lines each
- Skills: 150-300 lines typical
- If a skill needs >350 lines, it's trying to do too much
- Prefer tables and code over prose (higher information density)
- Avoid volatile content in AGENTS.md (timestamps, changing metrics) — breaks cache-friendly reuse
- Design subagent tasks for single-turn completion to prevent context accumulation
- See `references/workflow/context-efficiency` for full patterns

## Multi-Agent Orchestration

### Task Tool Patterns

```markdown
Use the Task tool to perform these searches in parallel:

1. Search for all uses of `deprecated_function` in src/
2. Search for the migration guide in docs/
3. Check the changelog for when it was deprecated
```

### Agent Specialization

| Agent Role | Task | Context Needed |
|---|---|---|
| Researcher | Find all usages of a pattern | Codebase access, grep |
| Implementer | Make specific code changes | File context, conventions |
| Reviewer | Validate changes against rules | Diff, style guide |
| Documenter | Update docs after changes | Changed files, doc templates |

### Background Agent Patterns
- Use `run_in_background` for long builds/tests while continuing other work
- Fire off parallel Task agents for independent research
- Collect results and synthesize in the main thread

### Orchestration Rules
- Main thread: decision-making, sequencing, user communication
- Subagents: independent, parallelizable work
- Never nest subagents more than one level deep
- Always specify what output format the subagent should return

## Permission Management

See `references/workflow/permission-management.md` for full settings hierarchy and JSON examples.

**Key rules:** minimal permissions by default, deny overrides allow, use glob patterns, `settings.local.json` for personal overrides.

## Gotchas

- **Context bloat**: Every line in AGENTS.md consumes context window. Ruthlessly prune. If Codex already knows it (general programming, language syntax), don't restate it.
- **Over-engineering skills**: A skill should solve a recurring problem. If you've only needed it once, it's not a skill -- it's a conversation. Wait for the third occurrence.
- **Stale instructions**: AGENTS.md that references deleted files, old conventions, or deprecated workflows actively misleads. Review quarterly.
- **Description mismatch**: If the skill description doesn't match what the skill actually does, Codex will invoke it at the wrong time or skip it when needed. Test trigger phrases.
- **Skill overlap**: Two skills with similar descriptions cause unpredictable invocation. Deduplicate or make descriptions clearly distinct.
- **Command complexity**: If a command file exceeds 10 lines, the logic should be in a skill. Commands are routing, not logic.
- **Hook performance**: A slow hook on every file write makes the entire workflow painful. Profile hooks and keep them under 2 seconds.
- **Ignoring layer precedence**: Putting project-specific rules in global AGENTS.md means they apply to every project. Keep global truly global.
