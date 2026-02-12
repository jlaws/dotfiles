---
name: claude-code-meta-patterns
description: "Use when optimizing Claude Code workflows, designing CLAUDE.md files, creating custom skills and commands, configuring hooks, or orchestrating multi-agent patterns."
---

# Claude Code Meta-Patterns

## CLAUDE.md Optimization

### What to Include vs. Avoid

| Include | Avoid |
|---------|-------|
| Project-specific conventions (naming, patterns) | Generic programming advice Claude already knows |
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
- Always start with "Use when" -- this is the trigger phrase Claude matches on
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
- Commands in `.claude/commands/` for project, `~/.claude/commands/` for global
- No logic in commands; all logic lives in skills
- Description should make the command discoverable

## Hook Patterns

### Pre-Commit Hook

Validate before committing:

```json
// .claude/settings.json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash(git commit)",
        "hook": "lint-staged && npm test"
      }
    ]
  }
}
```

### Post-Tool Validation

Check results after tool execution:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hook": "eslint --fix ${file} && prettier --write ${file}"
      }
    ]
  }
}
```

### Hook Design Principles
- Hooks should be fast (<5s); slow hooks degrade the workflow
- Fail hooks loudly with clear error messages
- Use matchers to scope hooks narrowly (don't lint on every Bash call)
- Test hooks manually before relying on them

## Context Management

### Keeping Context Small

| Strategy | When | How |
|----------|------|-----|
| Link, don't inline | Docs > 50 lines | "See docs/architecture.md for details" |
| Scope CLAUDE.md | Always | Only include what affects daily coding |
| Prune stale instructions | Monthly | Remove anything that hasn't been relevant |
| Use directory-level CLAUDE.md | Large monorepos | Put module-specific rules in module dirs |

### When to Use Subagents (Task Tool)

| Use Subagent | Stay in Main Thread |
|---|---|
| Independent research task | Sequential dependent steps |
| Exploring a large codebase area | Editing a specific file |
| Generating boilerplate for N items | Making a single targeted change |
| Parallel investigation of alternatives | Simple question-answer |

### Context Budget Rules
- CLAUDE.md files: aim for <150 lines each
- Skills: 150-300 lines typical
- If a skill needs >350 lines, it's trying to do too much
- Prefer tables and code over prose (higher information density)

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

### Settings Hierarchy

```
/Library/.../managed-settings.json   # Enterprise (highest priority)
~/.claude/settings.json              # User global
project/.claude/settings.json        # Project
project/.claude/settings.local.json  # Local (gitignored)
```

### Common Permission Patterns

```json
// .claude/settings.json (project)
{
  "permissions": {
    "allow": [
      "Bash(npm run *)",
      "Bash(git *)",
      "Read",
      "Write(src/**)",
      "Edit(src/**)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Write(.env*)",
      "Bash(git push --force*)"
    ]
  }
}
```

### Permission Rules
- Default to minimal permissions, expand as needed
- Deny rules override allow rules
- Use glob patterns for path-based permissions
- `settings.local.json` for personal overrides (gitignored)

## Gotchas

- **Context bloat**: Every line in CLAUDE.md consumes context window. Ruthlessly prune. If Claude already knows it (general programming, language syntax), don't restate it.
- **Over-engineering skills**: A skill should solve a recurring problem. If you've only needed it once, it's not a skill -- it's a conversation. Wait for the third occurrence.
- **Stale instructions**: CLAUDE.md that references deleted files, old conventions, or deprecated workflows actively misleads. Review quarterly.
- **Description mismatch**: If the skill description doesn't match what the skill actually does, Claude will invoke it at the wrong time or skip it when needed. Test trigger phrases.
- **Skill overlap**: Two skills with similar descriptions cause unpredictable invocation. Deduplicate or make descriptions clearly distinct.
- **Command complexity**: If a command file exceeds 10 lines, the logic should be in a skill. Commands are routing, not logic.
- **Hook performance**: A slow hook on every file write makes the entire workflow painful. Profile hooks and keep them under 2 seconds.
- **Ignoring layer precedence**: Putting project-specific rules in global CLAUDE.md means they apply to every project. Keep global truly global.
